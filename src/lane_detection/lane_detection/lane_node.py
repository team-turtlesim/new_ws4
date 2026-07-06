"""차선 인지 노드 (순수 인지).

opencv_node 가 발행하는 엣지 영상(/opencv/image/edge)을 구독하여, 관심영역(ROI)
안에서 좌/우 차선을 검출하고 '그 순간'의 기하값(차선 중심, 횡오차, 진행방향
기울기, 신뢰도)을 LaneDetection 으로 발행한다.

이 노드는 인지(perception)만 담당한다: 픽셀에서 선을 뽑고 기하값을 계산할 뿐,
시간 평활(EMA)·데드밴드·클램프·미검출 시 값 유지 같은 '판단'은 하지 않는다.
그 판단은 interpret 노드가 LaneDetection 을 구독해 수행하고 LaneInfo 로 재발행한다.

파이프라인:
    엣지 영상 -> ROI 자르기 -> 행별 차선 픽셀 검출 -> 다항식 피팅/이상치 제거
    -> 단일선 병합 -> 차선폭 학습 -> raw offset/center/confidence 계산
    -> LaneDetection 발행 (+ 선택적 디버그 시각화 영상 발행)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from interface.msg import LaneDetection


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # --- ROS parameters -------------------------------------------------
        self.declare_parameter('edge_topic', '/opencv/image/edge')
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('debug_topic', '/lane_detection/image/debug')
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('num_scan_rows', 12)     # ROI 안에서 스캔할 가로줄 개수
        self.declare_parameter('min_detect_rows', 3)    # 차선으로 인정할 최소 검출 줄 수
        # 2026-07-05 실측: 이 트랙 차선폭 ≈ 178px(0.556×320). 기본/범위를 실측에 맞춤.
        self.declare_parameter('default_lane_width_ratio', 0.556)  # 초기 차선폭(이미지폭 대비)
        # 학습된 차선폭(px)을 이미지폭 대비 이 범위로 clamp. 단일차선 추종 시 half 가
        # 과도하게 커져(=반대편으로 overshoot) 반대 차선을 넘는 것을 좌우 대칭으로 방지.
        self.declare_parameter('lane_width_min_ratio', 0.42)
        self.declare_parameter('lane_width_max_ratio', 0.62)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_log', False)  # lane_width/검출상태 진단 로그

        edge_topic = str(self.get_parameter('edge_topic').value)
        detection_topic = str(self.get_parameter('detection_topic').value)
        debug_topic = str(self.get_parameter('debug_topic').value)
        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.num_scan_rows = max(2, int(self.get_parameter('num_scan_rows').value))
        self.min_detect_rows = max(1, int(self.get_parameter('min_detect_rows').value))
        self.default_lane_width_ratio = float(self.get_parameter('default_lane_width_ratio').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be in range [0, 100]')
        self.debug_image = bool(self.get_parameter('debug_image').value)

        # --- vehicle_config.yaml 에서 ROI 읽기 ------------------------------
        self.roi_top, self.roi_left = self.load_roi()
        # config 값을 기본값으로 하되, 실시간 튜닝을 위해 ROS 파라미터로도 노출.
        # detect_lane / publish_debug 는 매 프레임 파라미터를 다시 읽으므로
        # `ros2 param set /lane_detection_node roi_top N` 으로 라이브 조정 가능.
        self.declare_parameter('roi_top', int(self.roi_top))
        self.declare_parameter('roi_left', int(self.roi_left))
        # 라인 피팅 이상치 제거 임계값(px). 이 값보다 선에서 멀면 사물로 보고 버림.
        self.declare_parameter('line_fit_outlier_px', 12.0)
        # 피팅 차수. 근거리 밴드엔 1(직선)이 안정적. 2는 과적합→가짜곡선 요동.
        self.declare_parameter('line_fit_degree', 1)
        # 단일선 판별: 좌x·우x 간격이 (차선폭 * 이 비율)보다 작으면 사실 같은 선
        # 하나가 중심을 가로질러 좌/우로 잘린 것으로 보고 하나의 차선으로 병합.
        self.declare_parameter('single_line_gap_ratio', 0.55)
        # --- 좌/우 분류(클러스터 추적)용 ---
        # cluster_gap_px: 한 행에서 이 간격(px) 이하로 붙은 엣지 픽셀은 한 선으로 묶음.
        # (한 선의 Canny 양쪽 엣지는 붙여서 1개로, 서로 다른 두 차선은 분리)
        self.declare_parameter('cluster_gap_px', 30.0)
        # min_lane_sep_ratio: 근거리 씨앗에서 두 클러스터가 (이 비율*이미지폭) 이상
        # 떨어져야 두 차선으로 인정. 미만이면 단일선(중앙 걸침)으로 취급 → 유령선 방지.
        self.declare_parameter('min_lane_sep_ratio', 0.2)
        # track_tol_px: 인접 스캔행 간 같은 차선으로 매칭할 최대 x 이동(px).
        self.declare_parameter('track_tol_px', 40.0)

        # --- 내부 상태 ------------------------------------------------------
        # 차선폭(px)은 기하 상태라 인지에 둔다. 양쪽 검출 시 EMA로 학습해
        # 한쪽만 보일 때 반대편 차선 위치를 추정하는 데 쓴다. (시간 평활 아님)
        self.lane_width_px = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            edge_topic,
            self.image_callback,
            image_qos,
        )
        self.detection_pub = self.create_publisher(LaneDetection, detection_topic, 10)
        self.debug_pub = None
        if self.debug_image:
            self.debug_pub = self.create_publisher(CompressedImage, debug_topic, image_qos)

        self.get_logger().info(
            'lane_detection node started (perception only):\n'
            f'  edge_topic={edge_topic}\n'
            f'  detection_topic={detection_topic}\n'
            f'  roi_top={self.roi_top}, roi_left={self.roi_left}\n'
            f'  num_scan_rows={self.num_scan_rows}, min_detect_rows={self.min_detect_rows}\n'
            f'  debug_image={self.debug_image}'
        )

    # ------------------------------------------------------------------ config
    def load_roi(self):
        roi_top, roi_left = 0, 0
        if not os.path.exists(self.vehicle_config_file):
            self.get_logger().warning(
                f'vehicle config not found ({self.vehicle_config_file}); ROI defaults 0.'
            )
            return roi_top, roi_left
        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as stream:
                config_data = yaml.safe_load(stream) or {}
        except Exception as exc:
            self.get_logger().warning(f'Failed to read vehicle config: {exc}')
            return roi_top, roi_left
        roi_top = int(config_data.get('ROI_TOP', 0))
        roi_left = int(config_data.get('ROI_LEFT', 0))
        return roi_top, roi_left

    # ------------------------------------------------------------------ decode
    def decode_edge(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        edge = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if edge is None:
            self.get_logger().warning('Failed to decode edge image')
        return edge

    # --------------------------------------------------------------- detection
    def detect_lane(self, edge):
        """ROI 안에서 행별로 좌/우 차선 x좌표를 찾아 '그 순간'의 차선 중심과
        offset/confidence 를 계산한다. 시간 평활은 하지 않는다."""
        height, width = edge.shape
        center_x = width / 2.0
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        roi_left = min(max(int(self.get_parameter('roi_left').value), 0), width - 1)

        if self.lane_width_px is None:
            self.lane_width_px = self.default_lane_width_ratio * width

        scan_ys = np.linspace(roi_top, height - 1, self.num_scan_rows).astype(int)

        # 좌/우 차선 점 검출: 화면 중심으로 자르지 않고, 행별 엣지 픽셀을 '선(클러스터)'
        # 으로 묶은 뒤 근거리(하단)에서 차선 수를 확정하고 위로 추적한다. 중앙 근처에
        # 걸친 한 개의 선이 좌/우 두 개로 쪼개지는 오분류(유령 반대선)를 막는다.
        left_raw, right_raw = self.scan_lanes(edge, scan_ys, roi_left, center_x, width)

        # 검출점에 다항식(곡선)을 피팅해 선에서 벗어난 엉뚱한 사물 점(이상치)을
        # 걸러내고, 차선을 정교한 곡선으로 표현한다.
        left_pts, left_poly = self.fit_and_filter(left_raw)
        right_pts, right_poly = self.fit_and_filter(right_raw)

        left_detected = len(left_pts) >= self.min_detect_rows
        right_detected = len(right_pts) >= self.min_detect_rows
        left_x = float(np.median([x for _, x in left_pts])) if left_detected else None
        right_x = float(np.median([x for _, x in right_pts])) if right_detected else None

        # --- 단일선 판별 (곡선에서 한 선이 중심을 가로질러 좌/우로 잘리는 문제) ---
        # 좌x·우x 간격이 실제 차선폭보다 훨씬 작으면 둘은 같은 선. 하나의 차선으로
        # 병합하고, 근거리(맨 아래) 위치가 중심의 어느 쪽인지로 좌/우를 판정한다.
        if left_detected and right_detected:
            ref_width = self.lane_width_px if self.lane_width_px else float(width)
            gap_ratio = float(self.get_parameter('single_line_gap_ratio').value)
            if (right_x - left_x) < gap_ratio * ref_width:
                all_pts = left_pts + right_pts
                _, x_near = max(all_pts, key=lambda p: p[0])  # 가장 아래(근거리) 점
                line_x = float(np.median([x for _, x in all_pts]))
                line_poly = left_poly if left_poly is not None else right_poly
                if x_near < center_x:      # 근거리에서 중심 왼쪽 -> 좌차선
                    left_detected, right_detected = True, False
                    left_x, right_x = line_x, None
                    left_pts, right_pts = all_pts, []
                    left_poly, right_poly = line_poly, None
                else:                       # 근거리에서 중심 오른쪽 -> 우차선
                    left_detected, right_detected = False, True
                    left_x, right_x = None, line_x
                    left_pts, right_pts = [], all_pts
                    left_poly, right_poly = None, line_poly

        # 필터된 점으로 per-row 차선중심 재구성 (단일선 병합 반영)
        left_map = {y: x for y, x in left_pts}
        right_map = {y: x for y, x in right_pts}
        center_pts = []
        for y in scan_ys:
            y = int(y)
            lx = left_map.get(y)
            rx = right_map.get(y)
            if lx is not None and rx is not None:
                center_pts.append((y, (lx + rx) / 2.0))
            elif lx is not None:
                center_pts.append((y, lx + self.lane_width_px / 2.0))
            elif rx is not None:
                center_pts.append((y, rx - self.lane_width_px / 2.0))

        # 양쪽 검출 시 차선폭 학습(EMA) — 기하 상태 추정(시간 평활 아님)
        if left_detected and right_detected and right_x > left_x:
            self.lane_width_px = 0.8 * self.lane_width_px + 0.2 * (right_x - left_x)

        # 차선폭을 안전 범위로 clamp -> 단일차선 half overshoot(반대선 침범) 방지.
        w_min = float(self.get_parameter('lane_width_min_ratio').value) * width
        w_max = float(self.get_parameter('lane_width_max_ratio').value) * width
        if w_max > w_min:
            self.lane_width_px = float(np.clip(self.lane_width_px, w_min, w_max))

        half = self.lane_width_px / 2.0
        if left_detected and right_detected:
            lane_center = (left_x + right_x) / 2.0
        elif left_detected:
            lane_center = left_x + half
        elif right_detected:
            lane_center = right_x - half
        else:
            lane_center = None

        detected_rows = len(center_pts)
        confidence = detected_rows / float(self.num_scan_rows)

        # raw offset: 그 순간의 정규화 횡오차. 미검출 시엔 0(=값 유지는 interpret 담당).
        if lane_center is not None:
            raw_offset = (lane_center - center_x) / (width / 2.0)
            raw_offset = float(np.clip(raw_offset, -1.0, 1.0))
        else:
            raw_offset = 0.0
            confidence = 0.0  # 완전 미검출: 신뢰도 0

        return {
            'raw_offset': raw_offset,
            'left_detected': left_detected,
            'right_detected': right_detected,
            'confidence': float(np.clip(confidence, 0.0, 1.0)),
            'lane_center': lane_center,
            'center_x': center_x,
            'image_width': int(width),
            'image_height': int(height),
            'left_pts': left_pts,
            'right_pts': right_pts,
            'left_poly': left_poly,
            'right_poly': right_poly,
        }

    def row_clusters(self, edge_row, roi_left, cluster_gap):
        """한 행의 엣지 픽셀을 x 간격 기준으로 묶어 클러스터 목록을 만든다.
        각 클러스터 = (mean_x, min_x, max_x). x(mean) 오름차순 정렬."""
        xs = np.where(edge_row[roi_left:] > 0)[0]
        if xs.size == 0:
            return []
        xs = np.sort(xs + roi_left)
        if xs.size == 1:
            x = int(xs[0])
            return [(float(x), x, x)]
        splits = np.where(np.diff(xs) > cluster_gap)[0]
        groups = np.split(xs, splits + 1)
        clusters = [(float(g.mean()), int(g.min()), int(g.max())) for g in groups]
        clusters.sort(key=lambda c: c[0])
        return clusters

    def scan_lanes(self, edge, scan_ys, roi_left, center_x, width):
        """행별 클러스터를 근거리(하단)→원거리(상단)로 추적해 좌/우 차선 점열을 만든다.

        목표: (1) 두 선이 있으면 둘 다 잡아 '두 선 사이 중앙'을 유지(정상 동작),
              (2) 한 선만 있으면(중앙에 걸쳐도) 유령 반대선을 만들지 않는다.

        - 매 행: 먼저 기존 좌/우 차선을 가장 가까운 클러스터에 track_tol 내에서
          매칭·갱신. 그다음 아직 없는 차선을 '충분히 떨어진(≥min_lane_sep)'
          미사용 클러스터에서 새로 시작한다 → 두 번째 선이 위쪽에서 늦게 나타나도
          받아들이되(정상 두 선 복원), 단일선은 행마다 클러스터가 하나뿐이라
          먼 미사용 클러스터가 없어 유령선이 생기지 않는다.
        - 좌차선 점은 안쪽 엣지(=오른쪽=max_x), 우차선은 안쪽(=왼쪽=min_x)을 기록해
          기존 캘리브레이션(차선폭/half) 관례를 유지한다."""
        cluster_gap = float(self.get_parameter('cluster_gap_px').value)
        track_tol = float(self.get_parameter('track_tol_px').value)
        min_lane_sep = float(self.get_parameter('min_lane_sep_ratio').value) * width

        left_raw, right_raw = [], []
        left_ref, right_ref = None, None  # 각 차선의 직전 행 mean x (추적 기준)

        def nearest_unused(ref, means, used):
            cand = [(abs(means[k] - ref), k) for k in range(len(means)) if k not in used]
            return min(cand)[1] if cand else None

        for y in sorted((int(v) for v in scan_ys), reverse=True):  # 근거리부터
            clusters = self.row_clusters(edge[y], roi_left, cluster_gap)
            if not clusters:
                continue
            means = [c[0] for c in clusters]
            used = set()

            # 1) 기존 차선 추적: 가장 가까운 미사용 클러스터를 tol 내에서 매칭
            if left_ref is not None:
                j = nearest_unused(left_ref, means, used)
                if j is not None and abs(means[j] - left_ref) <= track_tol:
                    left_ref = means[j]
                    left_raw.append((y, clusters[j][2]))  # 좌 안쪽 엣지 = max_x
                    used.add(j)
            if right_ref is not None:
                j = nearest_unused(right_ref, means, used)
                if j is not None and abs(means[j] - right_ref) <= track_tol:
                    right_ref = means[j]
                    right_raw.append((y, clusters[j][1]))  # 우 안쪽 엣지 = min_x
                    used.add(j)

            remaining = [k for k in range(len(clusters)) if k not in used]

            # 2) 아직 없는 차선을 '충분히 떨어진' 미사용 클러스터에서 시작
            if left_ref is None and right_ref is None:
                if len(remaining) >= 2 and \
                        (means[remaining[-1]] - means[remaining[0]]) >= min_lane_sep:
                    # 두 선 동시 씨앗 (최좌=좌, 최우=우)
                    a, b = remaining[0], remaining[-1]
                    left_ref, right_ref = means[a], means[b]
                    left_raw.append((y, clusters[a][2]))
                    right_raw.append((y, clusters[b][1]))
                elif remaining:
                    # 단일선(또는 붙은 덩어리): 한 덩어리로 보고 화면 중심 기준 한쪽만
                    all_min = min(clusters[k][1] for k in remaining)
                    all_max = max(clusters[k][2] for k in remaining)
                    m = 0.5 * (all_min + all_max)
                    if m < center_x:
                        left_ref = m
                        left_raw.append((y, all_max))
                    else:
                        right_ref = m
                        right_raw.append((y, all_min))
            elif left_ref is None:
                # 우차선만 있음 → 우차선보다 min_lane_sep 이상 왼쪽인 클러스터로 좌차선 시작
                cands = [k for k in remaining if right_ref - means[k] >= min_lane_sep]
                if cands:
                    k = min(cands, key=lambda k: means[k])
                    left_ref = means[k]
                    left_raw.append((y, clusters[k][2]))
            elif right_ref is None:
                # 좌차선만 있음 → 좌차선보다 min_lane_sep 이상 오른쪽인 클러스터로 우차선 시작
                cands = [k for k in remaining if means[k] - left_ref >= min_lane_sep]
                if cands:
                    k = max(cands, key=lambda k: means[k])
                    right_ref = means[k]
                    right_raw.append((y, clusters[k][1]))

        left_raw.sort()
        right_raw.sort()
        return left_raw, right_raw

    def fit_degree(self, n_points):
        """요청 차수를 파라미터에서 읽되, 점 수로 상한을 둔다(차수 = 점수-1 이하).
        근거리 밴드엔 기본 1차(직선)가 안정적."""
        req = int(self.get_parameter('line_fit_degree').value)
        return max(1, min(req, n_points - 1))

    def fit_and_filter(self, pts):
        """검출점들에 다항식 x=f(y)를 피팅(차선은 수직에 가까움)해 이상치를
        제거하고 (필터된 점 리스트, np.poly1d 또는 None)을 반환한다.
        점이 적으면 그대로 반환. 차수는 line_fit_degree 파라미터(기본 1차)."""
        if len(pts) < 3:
            return list(pts), None
        ys = np.array([p[0] for p in pts], dtype=np.float64)
        xs = np.array([p[1] for p in pts], dtype=np.float64)
        try:
            poly = np.poly1d(np.polyfit(ys, xs, self.fit_degree(len(pts))))
        except Exception:
            return list(pts), None

        resid = np.abs(xs - poly(ys))
        thresh = max(
            float(self.get_parameter('line_fit_outlier_px').value),
            2.5 * float(np.std(resid)),
        )
        keep = resid <= thresh
        if keep.all() or keep.sum() < 2:
            return list(pts), poly

        # 이상치 제거 후 1회 재피팅으로 선을 더 정교화
        ys2, xs2 = ys[keep], xs[keep]
        try:
            degree2 = self.fit_degree(int(ys2.size))
            poly = np.poly1d(np.polyfit(ys2, xs2, degree2))
        except Exception:
            pass
        filtered = [(int(y), int(x)) for y, x in zip(ys2, xs2)]
        return filtered, poly

    # ------------------------------------------------------------------ callbk
    def image_callback(self, msg: CompressedImage):
        edge = self.decode_edge(msg)
        if edge is None:
            return

        result = self.detect_lane(edge)

        if bool(self.get_parameter('debug_log').value):
            lc = result['lane_center']
            self.get_logger().info(
                f"lane_width_px={self.lane_width_px:.0f} "
                f"L={int(result['left_detected'])} R={int(result['right_detected'])} "
                f"lane_center={('%.0f' % lc) if lc is not None else 'None'} "
                f"raw_offset={result['raw_offset']:+.3f} conf={result['confidence']:.2f}",
                throttle_duration_sec=0.5,
            )

        detection = LaneDetection()
        detection.header.stamp = msg.header.stamp
        detection.header.frame_id = 'lane_detection'
        detection.image_width = result['image_width']
        detection.image_height = result['image_height']
        detection.center_x = float(result['center_x'])
        detection.lane_center_px = (
            float(result['lane_center']) if result['lane_center'] is not None else -1.0
        )
        detection.raw_offset = result['raw_offset']
        detection.left_detected = result['left_detected']
        detection.right_detected = result['right_detected']
        detection.confidence = result['confidence']
        self.detection_pub.publish(detection)

        if self.debug_pub is not None:
            self.publish_debug(edge, result, msg)

    # ------------------------------------------------------------------- debug
    def publish_debug(self, edge, result, source_msg: CompressedImage):
        canvas = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        height, width = edge.shape
        center_x = int(result['center_x'])

        # ROI 상단 경계선(노랑)
        roi_top = min(max(int(self.get_parameter('roi_top').value), 0), height - 1)
        cv2.line(canvas, (0, roi_top), (width, roi_top), (0, 255, 255), 1)
        # 이미지 중심선(흰색)
        cv2.line(canvas, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        # 좌/우 차선 검출점(좌=빨강, 우=파랑) — 참고용 작은 점
        for y, x in result['left_pts']:
            cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)
        for y, x in result['right_pts']:
            cv2.circle(canvas, (x, y), 1, (255, 0, 0), -1)
        # 피팅된 차선 곡선(좌=빨강, 우=파랑) — 정교한 실선
        for poly, color in ((result.get('left_poly'), (0, 0, 255)),
                            (result.get('right_poly'), (255, 0, 0))):
            if poly is None:
                continue
            ys = np.arange(roi_top, height)
            xs = np.clip(poly(ys), 0, width - 1).astype(np.int32)
            pts_line = np.stack([xs, ys.astype(np.int32)], axis=1)
            cv2.polylines(canvas, [pts_line], False, color, 2)
        # 차선 중심선(초록)
        if result['lane_center'] is not None:
            lc = int(result['lane_center'])
            cv2.line(canvas, (lc, roi_top), (lc, height), (0, 255, 0), 2)

        ok, encoded = cv2.imencode(
            '.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = 'lane_detection_debug'
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
