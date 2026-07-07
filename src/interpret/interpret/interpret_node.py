"""차선 판단+제어결정 노드 (interpret).

lane_detection 이 발행하는 '그 순간'의 인지값(LaneDetection, /lane/detection)을
구독하여, 한 콜백 안에서 두 가지를 연속으로 수행한다:

  1) 판단(시간필터/안전): offset EMA 저역통과. 결과를 LaneInfo(/lane_info)로도
     발행(디버그/rosbag 용; 현재 런타임 구독자는 없음).
  2) 제어결정(PID): offset 횡오차 PID 를 돌려 조향/스로틀을 계산하고
     Control(/control)로 발행. 하드웨어는 안 만진다(그건 control_node).

왜 인지(lane_detection)와 이 노드를 나누고, 예전의 lane_follow(제어)를 여기로
합쳤나:
  - 인지 = "지금 프레임에 선이 어디 있나"(기하). 그것만 lane_detection 이 한다.
  - 판단+제어결정 = "그걸 얼마나 믿고, 얼마나 꺾을까". 시간적 맥락이 필요한 이
    둘을 한 노드에 모아 한 콜백에서 처리한다.
  - 이렇게 하면 '프레임 도착 → 즉시 조향명령'이 되어(이벤트구동) 예전 lane_follow
    의 고정 20Hz 타이머가 만들던 최대 ~50ms 위상지연과 /lane_info 홉이 사라진다.

제어 방식(offset 전용):
  - 2026-07-06: heading(진행방향 기울기)은 이 셋업에서 주행 중 신뢰불가로 판명
    (종횡비 640x480->320x160 세로 1.5배 압축으로 기울기 증폭 + 점선 중앙선/비대칭
    검출 + ROI 외삽으로 직진에서도 heading 이 ±0.5 스파이크). 그래서 heading 기반
    곡선제어(선행조향/curve_bias)를 전부 걷어내고 offset 만으로 제어한다.
  - 직진: 순수 offset=0 중앙추종(PID). 곡선: offset 이 커지면(바깥 밀림) kp 를
    올리고(반응형) 감속해 라인을 유지. 곡선 감지도 |offset| 기반.

안전(페일세이프):
  - 이벤트구동이라 프레임이 끊기면 발행이 멈춘다. "명령이 끊기면 중립+정지"는
    control_node 의 stale 워치독이 담당한다.
  - 차선 신뢰도가 낮으면(lane lost) 스로틀을 0 으로 램프다운한다.
  - cruise_throttle 기본 0.0: 첫 기동엔 조향만 계산하고 차는 안 움직인다.
"""

import os
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from interface.msg import Control, DetectionArray, LaneDetection, LaneInfo
from std_msgs.msg import Bool


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def smoothstep(x, lo, hi):
    """lo~hi 구간을 0~1 로 S자 보간(양끝 기울기 0 -> 경계에서 게인이 부드럽게).
    x<=lo -> 0, x>=hi -> 1. 하드 분기 대신 연속 블렌딩용."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


class Judgment:
    """판단(Decision): 인지(차선/YOLO/ArUco)를 시간필터·신뢰·중재해 '목표'를 만든다.
    상태(필터값·인지 플래그)를 소유하고 파라미터/시계는 노드에서 읽는다. 제어(PID)는
    전혀 모른다 — offset 안정화 + 정지/감속 판정만 한다. (동작은 기존과 동일; 코드 조직만)"""

    def __init__(self, node):
        self.node = node
        self.ema_alpha = node.ema_alpha            # 노드가 읽고 검증한 값 그대로
        self.offset_filtered = 0.0                 # 미검출 시 마지막 값 유지
        # YOLO/ArUco 연동 상태(라벨/임계값 등은 콜백에서 매번 param 재읽기 → 라이브 튜닝)
        self.yolo_enabled = bool(node.get_parameter('yolo_enabled').value)
        self.yolo_stop = False
        self.yolo_slow = False
        self.last_yolo_time = None
        self.aruco_enabled = bool(node.get_parameter('aruco_enabled').value)
        self.aruco_stop = False
        self.last_aruco_time = None

    # --- 시간필터 ---
    def filter_offset(self, raw_offset, detected, single_line):
        """검출 시 EMA 저역통과, 미검출 시 마지막 필터값 유지.
        단독선일 때는 offset 을 축소·클램프해 신뢰도를 낮춘다(과대/요동 억제)."""
        gp = self.node.get_parameter
        if single_line:
            scale = float(gp('single_line_offset_scale').value)
            limit = float(gp('single_line_offset_limit').value)
            raw_offset = max(-limit, min(limit, raw_offset * scale))
        if detected:
            self.offset_filtered = (
                self.ema_alpha * raw_offset
                + (1.0 - self.ema_alpha) * self.offset_filtered
            )
        # 미검출: self.offset_filtered 를 직전 값 그대로 유지
        return float(max(-1.0, min(1.0, self.offset_filtered)))

    # --- 인지 플래그 갱신(구독 콜백에서 호출) ---
    def on_yolo(self, msg):
        """YOLO 검출을 받아 정지/감속 플래그를 갱신. 임계값·라벨은 매번 param 재읽기."""
        self.last_yolo_time = self.node.get_clock().now()
        gp = self.node.get_parameter
        self.yolo_enabled = bool(gp('yolo_enabled').value)
        if not self.yolo_enabled:
            self.yolo_stop = False
            self.yolo_slow = False
            return
        min_conf = float(gp('yolo_min_confidence').value)
        min_area = float(gp('yolo_min_box_area_ratio').value)
        stop_labels = set(gp('yolo_stop_labels').value)
        slow_labels = set(gp('yolo_slow_labels').value)
        img_area = float(max(1, int(msg.image_width) * int(msg.image_height)))
        stop = False
        slow = False
        for d in msg.detections:
            if float(d.confidence) < min_conf:
                continue
            # 박스 면적비 = 근접 프록시. 작은(먼) 검출은 무시해 조기·오반응 방지.
            if (float(d.width) * float(d.height)) / img_area < min_area:
                continue
            if d.label in stop_labels:
                stop = True
            elif d.label in slow_labels:
                slow = True
        self.yolo_stop = stop
        self.yolo_slow = slow

    def on_aruco(self, msg):
        """aruco_node 정지신호(/aruco_stop, 이미 디바운스됨)를 받아 플래그 갱신."""
        self.last_aruco_time = self.node.get_clock().now()
        self.aruco_enabled = bool(self.node.get_parameter('aruco_enabled').value)
        self.aruco_stop = bool(msg.data) if self.aruco_enabled else False

    # --- 인지 중재: 정지/감속 판정 ---
    def perception_stop_slow(self):
        """YOLO/ArUco 인지를 종합해 (stop, slow) 판정. 각 소스는 stale(노드 사망 등)이면
        무시한다(죽은 인지가 브레이크를 영구히 잡지 않게)."""
        now = self.node.get_clock().now()
        gp = self.node.get_parameter
        stop = False
        slow = False
        if self.yolo_enabled and self.last_yolo_time is not None:
            age = (now - self.last_yolo_time).nanoseconds * 1e-9
            if age <= float(gp('yolo_stop_timeout_sec').value):
                if self.yolo_stop:
                    stop = True
                elif self.yolo_slow:
                    slow = True
        if self.aruco_enabled and self.last_aruco_time is not None:
            age = (now - self.last_aruco_time).nanoseconds * 1e-9
            if age <= float(gp('aruco_stop_timeout_sec').value):
                if self.aruco_stop:
                    stop = True
        return stop, slow

    def debug_tag(self):
        """디버그 로그 접미사(원본과 동일 포맷)."""
        tag = ''
        if self.yolo_enabled:
            tag += ' yolo=' + ('STOP' if self.yolo_stop else 'slow' if self.yolo_slow else '-')
        if self.aruco_enabled and self.aruco_stop:
            tag += ' aruco=STOP'
        return tag


class Controller:
    """제어(Control law): 판단이 준 목표(offset + 정지/감속)를 PID 로 추종해 조향/스로틀
    명령을 계산한다. '무엇을/왜' 는 모른다 — 목표 추종만. 상태(적분·이전값·명령) 소유.
    (run_control 로직을 그대로 옮긴 것; 동작 동일. 발행/로그는 노드가 한다.)"""

    def __init__(self, node, steer_trim):
        self.node = node
        self.steer_trim = steer_trim
        self.prev_offset_for_d = 0.0
        self.prev_time = None                      # 미분/슬루 dt 계산용 (콜백 간 시간)
        self.throttle_cmd = 0.0
        self.steer_cmd_filtered = steer_trim       # EMA-smoothed 조향 출력
        self.was_low_conf = False                  # 직전 프레임 신뢰도 미달?
        self.integral = 0.0                        # offset 오차 적분(I 항)

    def step(self, offset, confidence, stop, slow):
        """offset PID + 스로틀 목표(정지/감속 반영) + 슬루 -> (steering, throttle, diag)."""
        gp = self.node.get_parameter
        now = self.node.get_clock().now()
        if self.prev_time is None:
            dt = 1.0 / 30.0   # 첫 프레임 가정치(카메라 ~30fps)
        else:
            dt = (now - self.prev_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / 30.0
        self.prev_time = now

        kp_straight = float(gp('kp_offset').value)
        kd = float(gp('kd_offset').value)
        ki = float(gp('ki_offset').value)
        i_limit = float(gp('i_limit').value)
        kp_curve = float(gp('kp_offset_curve').value)
        sched_off_lo = float(gp('sched_offset_lo').value)
        sched_off_hi = float(gp('sched_offset_hi').value)
        steer_limit = float(gp('steer_limit').value)
        steer_sign = float(gp('steer_sign').value)
        d_limit = float(gp('d_offset_limit').value)
        alpha = clip(float(gp('steer_smooth_alpha').value), 0.05, 1.0)
        min_conf = float(gp('min_confidence').value)

        low_conf = confidence < min_conf

        # 게인 스케줄링: |offset| 로 직진<->곡선 블렌딩(반응형).
        w = smoothstep(abs(offset), sched_off_lo, sched_off_hi)
        kp = lerp(kp_straight, kp_curve, w)

        error = offset  # 목표 = 차선중앙(offset=0)

        # 미분(클램프); 검출 복귀 프레임엔 리셋해 슬램 방지.
        if self.was_low_conf and not low_conf:
            self.prev_offset_for_d = offset
        d_offset = clip((offset - self.prev_offset_for_d) / dt, -d_limit, d_limit)
        self.prev_offset_for_d = offset
        self.was_low_conf = low_conf

        # 적분(anti-windup): 추종 중 & 전진 중일 때만 누적; 아니면 리셋.
        if low_conf or self.throttle_cmd <= 0.0:
            self.integral = 0.0
        elif ki > 0.0:
            self.integral += error * dt
            self.integral = clip(self.integral, -i_limit / ki, i_limit / ki)
        i_term = clip(ki * self.integral, -i_limit, i_limit)

        # PID 후 EMA 저역통과로 부드러운 출력.
        logical = kp * error + i_term + kd * d_offset
        logical = clip(logical, -steer_limit, steer_limit)
        steering_raw = clip(self.steer_trim + steer_sign * logical, -1.0, 1.0)
        self.steer_cmd_filtered = (
            alpha * steering_raw + (1.0 - alpha) * self.steer_cmd_filtered
        )
        steering = clip(self.steer_cmd_filtered, -1.0, 1.0)

        # 스로틀: lane lost 아니면 cruise; 곡선(w↑) 감속; 인지 정지/감속; 슬루 제한.
        cruise = float(gp('cruise_throttle').value)
        max_throttle = float(gp('max_throttle').value)
        slew = float(gp('throttle_slew_per_sec').value)
        curve_thr_scale = float(gp('curve_throttle_scale').value)
        throttle_scale = lerp(1.0, curve_thr_scale, w)
        target_throttle = 0.0 if low_conf else clip(cruise * throttle_scale, 0.0, max_throttle)
        # 판단이 준 정지/감속 반영 (원래 apply_perception_gate 의 적용부와 동일).
        if stop:
            target_throttle = 0.0
        elif slow:
            target_throttle = target_throttle * float(gp('yolo_slow_scale').value)

        step = slew * dt
        if target_throttle > self.throttle_cmd:
            self.throttle_cmd = min(self.throttle_cmd + step, target_throttle)
        else:
            self.throttle_cmd = max(self.throttle_cmd - step, target_throttle)

        return steering, self.throttle_cmd, {'i': i_term, 'd': d_offset, 'w': w, 'kp': kp}


class InterpretNode(Node):
    def __init__(self):
        super().__init__('interpret_node')

        # --- Topics / IO ----------------------------------------------------
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('lane_topic', '/lane_info')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())

        # --- 판단: offset 시간필터 -----------------------------------------
        # offset 저역통과 필터 계수(0~1, 클수록 민감/덜 평활).
        self.declare_parameter('ema_alpha', 0.4)
        # 단독선(한쪽만 검출)일 때 offset 축소/클램프 실험용 훅.
        # 2026-07-04 트랙 테스트: 축소(scale=0.5)하면 오히려 위치보정이 죽어 직진을
        # 못했다 → 기본값 1.0(비활성)으로 되돌림. 두 선 중앙잡기가 정상 동작이므로
        # 단독선 offset도 그대로 신뢰한다. 훅은 남겨두되 함부로 낮추지 말 것.
        self.declare_parameter('single_line_offset_scale', 1.0)
        self.declare_parameter('single_line_offset_limit', 1.0)

        # --- 제어결정: 횡오차 PID (offset 전용) ----------------------------
        # logical steer: 양수 = 물리적으로 우회전(LaneInfo 규약: +offset 이면 차선
        # 중심이 이미지 오른쪽 -> 차가 왼쪽으로 치우침 -> 우회전). steer_sign 이
        # 이를 서보 퍼센트 극성으로 매핑(-1.0 은 이전 차량 인계값, 트랙에서 검증).
        # 2026-07-06 주행 측정으로 확정한 게인. 이벤트구동 전환 후 남은 직진
        # 뱀주행(리밋사이클, ~0.5Hz)을 실측 튜닝으로 잡음:
        #   원본(kp0.45/kd0.12/ki0.2) offset std 0.110 -> 최종 std 0.060(-45%).
        self.declare_parameter('kp_offset', 0.25)   # 0.45->0.25 루프게인 축소(리밋사이클)
        self.declare_parameter('kd_offset', 0.16)   # 0.12->0.16 감쇠 강화(offset 깨끗해 여유)
        # 적분: 원래 정상상태 offset 제거용이나, 저속 위치루프에선 위상지연을 키워
        # 뱀주행을 되살린다(실측: ki=0.2/0.05 모두 std~0.11, ki=0 은 0.06). 그래서
        # 기본 0 으로 비활성. 정상상태 편향은 steer_trim 이 담당(측정 잔차 ±0.03).
        # 긴 직선에서 드리프트가 문제되면 i_limit 을 크게 낮춰(≤0.1) 소량만 재도입.
        self.declare_parameter('ki_offset', 0.0)
        self.declare_parameter('i_limit', 0.3)       # clamp on |ki*integral|
        self.declare_parameter('steer_limit', 0.7)   # max |logical steer|
        self.declare_parameter('steer_sign', -1.0)
        # steer_trim(직진/중립 조향값)은 vehicle_config 에서 읽는다.
        # --- 평활(직진 twitch 억제) ---
        # d_offset: 검출이 잠깐 끊겼다 복귀할 때 offset 이 튀어 미분이 폭발 -> 클램프.
        # steer EMA 로 최종 명령을 저역통과.
        self.declare_parameter('d_offset_limit', 2.0)     # clamp |d(offset)/dt|
        self.declare_parameter('steer_smooth_alpha', 0.30)  # 1.0 = no smoothing
        # --- 게인 스케줄링(직진<->곡선 연속 블렌딩, |offset| 기준) ---
        # w = smoothstep(|offset|, lo, hi): 0=중앙(직진) ~ 1=크게 벌어짐(곡선/이탈).
        # w 로 kp 를 직진값<->곡선값 사이에서 연속 보간. 하드 분기의 경계 튐을 피하려
        # S자 블렌딩. offset 이 커지면(곡선에서 바깥 밀림) kp 를 올려 강하게 복구.
        # 직진 뱀주행(|offset|~0.1)은 lo(0.3) 아래라 안 건드림 -> 직진 튜닝 유지.
        # heading 기반 곡선감지는 신뢰불가로 제거, offset(항상 유효)만으로 감지.
        self.declare_parameter('kp_offset_curve', 0.45)  # 곡선 kp(offset 교정 강화)
        self.declare_parameter('sched_offset_lo', 0.3)   # 이 |offset| 부터 복구게인 시작
        self.declare_parameter('sched_offset_hi', 0.6)   # 이 |offset| 에서 완전 곡선게인

        # --- 스로틀 ---------------------------------------------------------
        self.declare_parameter('cruise_throttle', 0.0)  # 0 => 조향부터 검증
        self.declare_parameter('max_throttle', 0.30)
        self.declare_parameter('throttle_slew_per_sec', 0.6)  # ramp rate
        # 곡선 감속: 조향 스케줄과 같은 w 로 throttle 을 줄인다("코너에서 브레이크").
        # 2026-07-06: kp0.45 로 강하게 걸어도 급곡선은 속도가 빠르면 못 버티고
        # 가장자리를 스침 -> 감속해야 라인 유지.
        # throttle = cruise × lerp(1.0, curve_throttle_scale, w). w=1 에서 이 비율로.
        self.declare_parameter('curve_throttle_scale', 0.9)

        # --- 페일세이프 -----------------------------------------------------
        self.declare_parameter('min_confidence', 0.2)   # 미만 -> lost 취급
        self.declare_parameter('debug_log', False)

        # --- YOLO 검출 연동(정지/감속) -------------------------------------
        # 기본 비활성(yolo_enabled=False): 켜기 전엔 /yolo/detections 를 구독만 하고
        # 스로틀에 전혀 관여하지 않아 기존 주행 동작이 100% 그대로다. 검증 후
        #   ros2 param set /interpret_node yolo_enabled true
        # 로 켠다. 조향은 건드리지 않는다(차선추종 유지) — 스로틀만 게이팅.
        self.declare_parameter('yolo_enabled', False)
        self.declare_parameter('yolo_detections_topic', '/yolo/detections')
        self.declare_parameter('yolo_min_confidence', 0.5)   # 미만 검출은 무시
        # 박스가 이미지의 이 비율 이상일 때만 반응(근접 프록시). 멀리 있는 작은 검출 무시.
        self.declare_parameter('yolo_min_box_area_ratio', 0.03)
        # 정지/감속시킬 클래스 (labels.txt 이름과 정확히 일치해야 함). 라이브 튜닝 가능.
        # 이 프로젝트 클래스 {green_light,left_sign,red_light,right_sign} 기준 기본 매핑:
        #   red_light -> 정지.  green_light -> 통과(어느 목록에도 없음).
        #   left_sign/right_sign -> 표시만(조향 영역이라 정지/감속엔 미연동).
        self.declare_parameter('yolo_stop_labels', ['red_light'])
        # 감속 전용 클래스는 없음. ['']=사실상 빈 목록(어떤 라벨과도 불일치). 필요시 추가.
        self.declare_parameter('yolo_slow_labels', [''])
        self.declare_parameter('yolo_slow_scale', 0.5)       # 감속 시 throttle 배율
        # 검출 끊김(노드 사망 등) 이 시간 초과면 게이팅 해제 — 죽은 인지가 브레이크를
        # 영구히 잡지 않도록. 실제 모션 페일세이프는 control_node stale 워치독이 담당.
        self.declare_parameter('yolo_stop_timeout_sec', 1.0)

        # --- ArUco 마커 연동(정지) -----------------------------------------
        # aruco_node 의 /aruco_stop(Bool, 이미 디바운스됨)을 구독해 True 면 정지시킨다.
        # 마커→정지의 '무엇을 정지대상으로 볼지'(target_marker_id) 는 aruco_node 가 판정하고,
        # interpret 은 그 신호를 받아 '그래서 스로틀을 0으로' = 판단의 나머지를 담당한다.
        # 조향엔 관여 안 함(차선추종 유지). 기본 활성(사용자 요청). off 하려면:
        #   ros2 param set /interpret_node aruco_enabled false
        self.declare_parameter('aruco_enabled', True)
        self.declare_parameter('aruco_stop_topic', '/aruco_stop')
        # 정지신호 끊김(노드 사망 등) 이 시간 초과면 게이팅 해제 — 죽은 인지가 브레이크를
        # 영구히 잡지 않도록.
        self.declare_parameter('aruco_stop_timeout_sec', 1.0)

        detection_topic = str(self.get_parameter('detection_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('ema_alpha must be in range (0, 1]')

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.steer_trim = self.load_steer_trim()

        # --- 판단/제어 분리 (동작 동일; 코드 조직만) ---
        # 판단(Judgment): 시간필터 + 인지 정지/감속 판정.  상태(필터값·인지 플래그) 소유.
        # 제어(Controller): 그 목표를 PID 로 추종.        상태(적분·이전값·명령) 소유.
        # 둘 다 파라미터/시계는 이 노드에서 읽는다. 한 콜백에서 이어 실행(이벤트구동 유지).
        self.judgment = Judgment(self)
        self.controller = Controller(self, self.steer_trim)

        self.subscription = self.create_subscription(
            LaneDetection,
            detection_topic,
            self.detection_callback,
            10,
        )
        self.lane_pub = self.create_publisher(LaneInfo, lane_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)

        # YOLO 검출 구독(정지/감속 게이팅용). 항상 구독하되 gate 는 yolo_enabled 로 제어.
        self.yolo_sub = self.create_subscription(
            DetectionArray,
            str(self.get_parameter('yolo_detections_topic').value),
            self.yolo_callback,
            10,
        )

        # ArUco 정지신호 구독(정지 게이팅용). 항상 구독하되 gate 는 aruco_enabled 로 제어.
        self.aruco_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('aruco_stop_topic').value),
            self.aruco_callback,
            10,
        )

        self.get_logger().info(
            'interpret node started (judgment + control law, offset-only):\n'
            f'  detection_topic={detection_topic}\n'
            f'  lane_topic={lane_topic}\n'
            f'  control_topic={control_topic}\n'
            f'  steer_trim={self.steer_trim} steer_sign='
            f'{float(self.get_parameter("steer_sign").value)}\n'
            f'  ema_alpha={self.ema_alpha}\n'
            f'  kp={float(self.get_parameter("kp_offset").value)}'
            f'(curve {float(self.get_parameter("kp_offset_curve").value)}) '
            f'kd={float(self.get_parameter("kd_offset").value)} '
            f'ki={float(self.get_parameter("ki_offset").value)}\n'
            f'  cruise_throttle={float(self.get_parameter("cruise_throttle").value)} '
            f'(0 => 조향만; 검증 후 param 으로 올릴 것)'
        )

    # ------------------------------------------------------------------ callbk
    def detection_callback(self, msg: LaneDetection):
        """프레임 도착 즉시: 판단(시간필터+정지/감속 판정) -> LaneInfo 발행 -> 제어(PID)
        -> Control 발행. 이벤트구동(한 콜백에서 판단→제어 이어 실행)."""
        detected = bool(msg.left_detected or msg.right_detected)
        # 단독선 = 좌/우 중 정확히 한쪽만 검출 (XOR)
        single_line = bool(msg.left_detected) != bool(msg.right_detected)

        # --- 판단: offset 시간필터 ---
        offset = self.judgment.filter_offset(float(msg.raw_offset), detected, single_line)
        confidence = float(msg.confidence)

        # 판단 결과를 LaneInfo 로도 발행(디버그/rosbag; 런타임 구독자는 없음).
        lane_info = LaneInfo()
        lane_info.header.stamp = msg.header.stamp
        lane_info.header.frame_id = 'interpret'
        lane_info.lane_offset = offset
        lane_info.left_detected = bool(msg.left_detected)
        lane_info.right_detected = bool(msg.right_detected)
        lane_info.confidence = confidence
        self.lane_pub.publish(lane_info)

        # --- 판단: 인지 정지/감속 판정 -> 제어(PID)로 목표 추종 -> Control 발행 ---
        stop, slow = self.judgment.perception_stop_slow()
        steering, throttle, diag = self.controller.step(offset, confidence, stop, slow)
        self.publish_control(steering, throttle)

        if bool(self.get_parameter('debug_log').value):
            self.get_logger().info(
                f'off={offset:+.3f} i={diag["i"]:+.3f} d={diag["d"]:+.3f} '
                f'conf={confidence:.2f} w={diag["w"]:.2f} kp={diag["kp"]:.2f} '
                f'-> steer={steering:+.3f} thr={throttle:.3f}{self.judgment.debug_tag()}'
            )

    def publish_control(self, steering, throttle):
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        self.control_pub.publish(msg)

    # --------------------------------------------------------- 인지 구독(판단 위임)
    # 구독 콜백은 판단(Judgment)으로 위임한다. 실제 정지/감속 판정과 필터는 Judgment 가,
    # PID 는 Controller 가 담당한다(위쪽 클래스). 노드는 배선/발행/로그만.
    def yolo_callback(self, msg: DetectionArray):
        self.judgment.on_yolo(msg)

    def aruco_callback(self, msg: Bool):
        self.judgment.on_aruco(msg)

    # ------------------------------------------------------------------ config
    def load_steer_trim(self):
        if not os.path.exists(self.vehicle_config_file):
            return 0.0
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(
                f'Failed to read vehicle config file {self.vehicle_config_file}: {exc}'
            )
            return 0.0
        return float(config_data.get('STEER_TRIM', 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = InterpretNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
