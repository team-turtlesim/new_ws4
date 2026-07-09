"""ArUco 마커 인지 노드 (car_ws 의 aruco_detector_node 를 new_ws 로 이식).

카메라 압축영상(/camera/image/compressed)을 구독해 ArUco 마커를 검출하고 발행한다:
  - /detected_marker_id (Int32)          : 검출된 마커 ID(첫 번째)
  - /aruco_stop (Bool)                   : 지정 ID(target_marker_id) 등장/소멸을 디바운스한 정지신호
  - /aruco/image/debug (CompressedImage) : 대시보드용 오버레이 (원본 imshow 대체 — 보드는 headless)

lane_detection/yolo 와 같은 '독립 인지 브랜치'로, 카메라 원본을 직접 소비한다.

── 원본(car_ws) 대비 이식 변경점 (3가지) ─────────────────────────────
  1) cv2.aruco 신 API(ArucoDetector, OpenCV 4.7+) → 이 보드 cv2 4.5.4 는 구 API 라
     getPredefinedDictionary + detectMarkers(함수) 사용. 두 API 모두 되게 shim 처리.
  2) cv_bridge 제거 → np.frombuffer + cv2.imdecode (new_ws 관례, 의존성 축소).
  3) cv2.imshow/waitKey(GUI) 제거 → /aruco/image/debug 로 발행 (보드에 디스플레이 없음).
그 외 여러분 로직(6X6_50 기본 + 백업 딕셔너리 교차검증 + 정지신호 디바운스)은 보존.

── 구조 메모 ────────────────────────────────────────────────────
target_marker_id → 정지신호(/aruco_stop) 판정은 사실 '판단'에 가깝다. 지금은 원본
동작을 살려 이 노드에 두지만, 나중에 판단(interpret) 연동 시 이 정지 결정은 interpret
로 옮겨 이 노드를 '순수 검출'로 남기는 게 아키텍처상 맞다(인지↔판단 분리).
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32


# 문자열 이름 -> cv2.aruco 사전 상수 (파라미터로 사전 선택용)
def _dict_const(name):
    return getattr(cv2.aruco, 'DICT_' + name)


def build_detector(dict_id):
    """cv2 버전에 무관하게 (mode, detector, dictionary, params) 를 만든다.
    - 4.7+ : ArucoDetector 클래스, detector.detectMarkers(img)
    - <4.7 : detectMarkers(img, dictionary, parameters=params) 함수  (이 보드)"""
    a = cv2.aruco
    if hasattr(a, 'getPredefinedDictionary'):
        dictionary = a.getPredefinedDictionary(dict_id)
    else:  # 아주 오래된 버전 대비
        dictionary = a.Dictionary_get(dict_id)
    if hasattr(a, 'ArucoDetector'):  # 신 API
        params = a.DetectorParameters()
        return ('new', a.ArucoDetector(dictionary, params), dictionary, params)
    # 구 API
    params = a.DetectorParameters_create()
    return ('old', None, dictionary, params)


def run_detect(entry, gray):
    """(corners, ids, rejected) 반환. entry = build_detector 결과."""
    mode, detector, dictionary, params = entry
    if mode == 'new':
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_node')

        # --- Topics ---------------------------------------------------------
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('marker_id_topic', '/detected_marker_id')
        self.declare_parameter('stop_topic', '/aruco_stop')
        self.declare_parameter('debug_topic', '/aruco/image/debug')

        # --- 검출 설정 ------------------------------------------------------
        # 기본 사전(원본과 동일 6X6_50). 실시간 교체: ros2 param 은 문자열, 아래 이름 사용.
        self.declare_parameter('aruco_dict', '6X6_50')
        # 교차검증(진단용): 기본 사전으로 못 찾으면 백업 사전들로 재시도해 '진짜 규격'을 알려줌.
        # 단, 마커가 안 보이는 매 프레임마다 사전 4~5개를 전부 다시 돌려 CPU 를 크게 먹는다.
        # 실측: 켜면 aruco 노드가 CPU 80% 점유 -> YOLO fps 40% 손실(1.96 vs 3.28).
        # 그래서 기본 OFF. 마커 규격을 모를 때만 잠깐 켜서 로그로 확인하고 다시 끈다.
        #   ros2 param set /aruco_node enable_crosscheck true   (라이브 토글 가능)
        self.declare_parameter('enable_crosscheck', False)
        self.declare_parameter('alt_dicts', ['6X6_250', '5X5_50', '5X5_250', '4X4_50'])

        # --- 정지신호(판단성) 디바운스 ------------------------------------
        self.declare_parameter('target_marker_id', 3)   # 이 ID만 장애물로 반응
        self.declare_parameter('stop_on_frames', 1)     # 등장: N프레임 연속 -> 즉시 정지
        self.declare_parameter('go_after_frames', 5)    # 소멸: N프레임 연속 미검출 -> 출발

        # --- 디버그 오버레이 ------------------------------------------------
        self.declare_parameter('debug_image', True)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('debug_log', True)

        image_topic = str(self.get_parameter('image_topic').value)
        self.debug_topic = str(self.get_parameter('debug_topic').value)
        self.debug_image = bool(self.get_parameter('debug_image').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_log = bool(self.get_parameter('debug_log').value)

        # --- 검출기 구성 ----------------------------------------------------
        primary_name = str(self.get_parameter('aruco_dict').value)
        self.primary = build_detector(_dict_const(primary_name))
        self.api_mode = self.primary[0]
        # 백업 사전 검출기는 항상 만들어 둔다(생성 비용은 시작 시 1회, 아주 쌈).
        # 실제로 '쓸지'는 매 프레임 enable_crosscheck 를 다시 읽어 결정 -> 라이브 토글 가능.
        self.alt = []
        for name in list(self.get_parameter('alt_dicts').value):
            try:
                self.alt.append((name, build_detector(_dict_const(str(name)))))
            except Exception as exc:
                self.get_logger().warning(f'alt dict {name} 무시: {exc}')

        # --- 디바운스 상태 --------------------------------------------------
        self.seen_count = 0
        self.notseen_count = 0
        self.stop_state = False

        # --- QoS: 카메라(RELIABLE)와 호환. 오버레이는 monitor(RELIABLE)와 맞춰 RELIABLE. ---
        cam_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                             reliability=ReliabilityPolicy.RELIABLE)
        overlay_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE)

        self.id_pub = self.create_publisher(
            Int32, str(self.get_parameter('marker_id_topic').value), 10)
        self.stop_pub = self.create_publisher(
            Bool, str(self.get_parameter('stop_topic').value), 10)
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, overlay_qos)

        self.subscription = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, cam_qos)

        self.get_logger().info(
            'aruco node started (cv2 %s, %s API):\n'
            '  image_topic=%s\n'
            '  marker_id_topic=%s  stop_topic=%s\n'
            '  debug_topic=%s\n'
            '  dict=%s  crosscheck=%s(%d)\n'
            '  target_marker_id=%s stop_on=%s go_after=%s' % (
                cv2.__version__, self.api_mode, image_topic,
                str(self.get_parameter('marker_id_topic').value),
                str(self.get_parameter('stop_topic').value),
                self.debug_topic if self.debug_image else '(disabled)',
                primary_name, bool(self.get_parameter('enable_crosscheck').value), len(self.alt),
                self.get_parameter('target_marker_id').value,
                self.get_parameter('stop_on_frames').value,
                self.get_parameter('go_after_frames').value,
            )
        )

    # ------------------------------------------------------------------ callbk
    def image_callback(self, msg: CompressedImage):
        frame = self.decode(msg)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)  # 조명 보정(원본과 동일)
        display = frame.copy() if self.debug_pub is not None else None

        # 1차: 기본 사전으로 검출
        corners, ids, rejected = run_detect(self.primary, equalized)

        # 정지신호(판단성) 디바운스 갱신 — 매 프레임
        self.update_stop_signal(ids)

        if ids is not None:
            marker_id = int(ids[0][0])
            self.publish_id(marker_id)
            if display is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids, borderColor=(0, 255, 0))
                cv2.putText(display, 'DETECTED  ID = %d' % marker_id, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            self.publish_overlay(display, msg)
            if self.debug_log:
                self.get_logger().info('마커 검출 ID=%d' % marker_id, throttle_duration_sec=0.5)
            return

        # 2차(교차검증, 기본 OFF): 매 프레임 param 을 다시 읽어 켜졌을 때만 백업 사전들로 재시도.
        # 켜면 사전 4~5개를 매 프레임 다 돌려 CPU 를 크게 먹으므로 규격 진단할 때만 잠깐 켤 것.
        if (bool(self.get_parameter('enable_crosscheck').value)
                and self.alt and rejected is not None and len(rejected) > 0):
            for name, entry in self.alt:
                a_corners, a_ids, _ = run_detect(entry, equalized)
                if a_ids is not None:
                    real_id = int(a_ids[0][0])
                    self.publish_id(real_id)
                    if display is not None:
                        cv2.aruco.drawDetectedMarkers(
                            display, a_corners, a_ids, borderColor=(0, 165, 255))
                        cv2.putText(display, '%s  ID=%d (규격 불일치)' % (name, real_id), (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                    self.publish_overlay(display, msg)
                    self.get_logger().warning(
                        '규격 불일치: 이 마커의 실제 사전은 [%s], ID=%d. '
                        'aruco_dict 파라미터를 이 규격으로 바꾸면 1차에서 잡힙니다.' % (name, real_id),
                        throttle_duration_sec=2.0)
                    return
            if display is not None:
                cv2.aruco.drawDetectedMarkers(display, rejected, borderColor=(0, 0, 255))

        # 미검출
        if display is not None:
            cv2.putText(display, 'NO MARKER', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        self.publish_overlay(display, msg)
        if self.debug_log:
            self.get_logger().info('유효한 ArUco 마커 없음', throttle_duration_sec=2.0)

    # ------------------------------------------------------------------ stop
    def update_stop_signal(self, ids):
        """지정 ID 마커의 등장/소멸을 디바운스해 /aruco_stop(Bool) 발행. 매 프레임 호출.
          - 등장 : stop_on_frames 연속 보이면 즉시 정지(True)   -> 장애물 늦게 반응 방지
          - 소멸 : go_after_frames 연속 안 보여야 출발(False)   -> 경계 깜빡임 방지"""
        target = int(self.get_parameter('target_marker_id').value)
        stop_on = int(self.get_parameter('stop_on_frames').value)
        go_after = int(self.get_parameter('go_after_frames').value)

        seen = ids is not None and target in [int(x) for x in ids.flatten()]
        if seen:
            self.seen_count += 1
            self.notseen_count = 0
        else:
            self.notseen_count += 1
            self.seen_count = 0

        if self.seen_count >= stop_on:
            self.stop_state = True
        elif self.notseen_count >= go_after:
            self.stop_state = False

        m = Bool()
        m.data = bool(self.stop_state)
        self.stop_pub.publish(m)

    # ------------------------------------------------------------------ io
    def decode(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('압축영상 디코드 실패')
        return frame

    def publish_id(self, marker_id):
        out = Int32()
        out.data = int(marker_id)
        self.id_pub.publish(out)

    def publish_overlay(self, display, src: CompressedImage):
        if self.debug_pub is None or display is None:
            return
        # 하단에 정지신호 상태 배너(원본 GUI 배너를 오버레이로)
        h = display.shape[0]
        if self.stop_state:
            cv2.putText(display, 'OBSTACLE -> STOP', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(display, 'CLEAR -> GO', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)
        ok, enc = cv2.imencode('.jpg', display, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = src.header.stamp
        out.header.frame_id = 'aruco_debug'
        out.format = 'jpeg'
        out.data = enc.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
