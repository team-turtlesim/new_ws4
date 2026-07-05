"""차선 판단 노드 (interpret).

lane_detection 이 발행하는 '그 순간'의 인지값(LaneDetection, /lane/detection)을
구독하여, 시간적 평활과 안전처리(판단)를 적용한 뒤 제어에 바로 쓸 수 있는
LaneInfo(/lane_info)로 재발행한다.

인지(lane_detection)와 판단(이 노드)을 노드 단위로 분리한 이유:
    - "지금 프레임에 선이 어디 있나"(기하) 와 "그걸 얼마나 믿고 부드럽게 쓸까"
      (시간·안전) 를 명확히 나눠 디버깅/튜닝을 한 곳으로 모은다.

판단 내용:
    - offset: EMA 저역통과 필터. 미검출(양쪽 모두 미검출) 시 마지막 값 유지.
    - heading: 신뢰도 낮으면 0으로 서서히 수렴(폭주 방지), 물리적 최대각으로
      클램프, 작은 각은 데드밴드로 직선 처리, 시간 EMA 평활.
"""

import rclpy
from rclpy.node import Node

from interface.msg import LaneDetection, LaneInfo


class InterpretNode(Node):
    def __init__(self):
        super().__init__('interpret_node')

        # --- ROS parameters -------------------------------------------------
        self.declare_parameter('detection_topic', '/lane/detection')
        self.declare_parameter('lane_topic', '/lane_info')
        # offset 저역통과 필터 계수(0~1, 클수록 민감/덜 평활).
        self.declare_parameter('ema_alpha', 0.4)
        # heading 안정화: 이 값(rad)보다 작은 기울기는 직선(0)으로 처리.
        # 2026-07-05 튜닝: 직진에서 heading 이 ±0.13 잔차로 흔들려 curve_bias 가
        # 반응→뱀주행. 0.05→0.15 로 올려 직진 흔들림은 죽이고 곡선만 통과시킴.
        self.declare_parameter('heading_deadband', 0.15)
        self.declare_parameter('heading_ema_alpha', 0.3)
        # heading 폭주 방지: 물리적으로 말이 되는 최대각(rad). 초과 시 클램프.
        self.declare_parameter('heading_limit', 0.45)
        # 이 신뢰도 미만이면 heading 을 신뢰하지 않고 0으로 서서히 수렴.
        self.declare_parameter('heading_min_conf', 0.5)
        # 단독선(한쪽만 검출)일 때 offset 축소/클램프 실험용 훅.
        # 2026-07-04 트랙 테스트: 축소(scale=0.5)하면 오히려 위치보정이 죽어 직진을
        # 못했다 → 기본값 1.0(비활성)으로 되돌림. 두 선 중앙잡기가 정상 동작이므로
        # 단독선 offset도 그대로 신뢰한다. 훅은 남겨두되 함부로 낮추지 말 것.
        self.declare_parameter('single_line_offset_scale', 1.0)
        self.declare_parameter('single_line_offset_limit', 1.0)

        detection_topic = str(self.get_parameter('detection_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError('ema_alpha must be in range (0, 1]')

        # --- 내부 상태 (판단용 시간필터 상태) --------------------------------
        self.offset_filtered = 0.0     # 필터링된 lane_offset (미검출 시 마지막 값 유지)
        self.heading_filtered = 0.0    # 시간 평활된 heading (rad)

        self.subscription = self.create_subscription(
            LaneDetection,
            detection_topic,
            self.detection_callback,
            10,
        )
        self.lane_pub = self.create_publisher(LaneInfo, lane_topic, 10)

        self.get_logger().info(
            'interpret node started (judgment):\n'
            f'  detection_topic={detection_topic}\n'
            f'  lane_topic={lane_topic}\n'
            f'  ema_alpha={self.ema_alpha}, '
            f'heading_deadband={float(self.get_parameter("heading_deadband").value)}, '
            f'heading_limit={float(self.get_parameter("heading_limit").value)}'
        )

    # ------------------------------------------------------------------ callbk
    def detection_callback(self, msg: LaneDetection):
        detected = bool(msg.left_detected or msg.right_detected)
        # 단독선 = 좌/우 중 정확히 한쪽만 검출 (XOR)
        single_line = bool(msg.left_detected) != bool(msg.right_detected)

        offset = self.filter_offset(float(msg.raw_offset), detected, single_line)
        heading = self.filter_heading(float(msg.raw_heading), float(msg.confidence))

        lane_info = LaneInfo()
        lane_info.header.stamp = msg.header.stamp
        lane_info.header.frame_id = 'interpret'
        lane_info.lane_offset = offset
        lane_info.heading_error = heading
        lane_info.curvature = 0.0
        lane_info.left_detected = bool(msg.left_detected)
        lane_info.right_detected = bool(msg.right_detected)
        lane_info.confidence = float(msg.confidence)
        self.lane_pub.publish(lane_info)

    # ------------------------------------------------------------------ filters
    def filter_offset(self, raw_offset, detected, single_line):
        """검출 시 EMA 저역통과, 미검출 시 마지막 필터값 유지.
        단독선일 때는 offset 을 축소·클램프해 신뢰도를 낮춘다(과대/요동 억제)."""
        if single_line:
            scale = float(self.get_parameter('single_line_offset_scale').value)
            limit = float(self.get_parameter('single_line_offset_limit').value)
            raw_offset = max(-limit, min(limit, raw_offset * scale))
        if detected:
            self.offset_filtered = (
                self.ema_alpha * raw_offset
                + (1.0 - self.ema_alpha) * self.offset_filtered
            )
        # 미검출: self.offset_filtered 를 직전 값 그대로 유지
        return float(max(-1.0, min(1.0, self.offset_filtered)))

    def filter_heading(self, raw_heading, confidence):
        """신뢰도/데드밴드/클램프/EMA 로 heading 을 안정화한다.
        신뢰도가 낮으면 기울기가 엉터리 -> 직선(0)으로 서서히 수렴시켜
        가짜 큰 각으로 조향이 튀는 것을 방지한다."""
        alpha = float(self.get_parameter('heading_ema_alpha').value)

        if confidence < float(self.get_parameter('heading_min_conf').value):
            self.heading_filtered *= (1.0 - alpha)
            return self.heading_filtered

        # 폭주 방지: 물리적으로 말이 되는 최대각으로 클램프
        limit = float(self.get_parameter('heading_limit').value)
        heading = max(-limit, min(limit, raw_heading))

        # 데드밴드: 작은 기울기는 직선으로
        deadband = float(self.get_parameter('heading_deadband').value)
        if abs(heading) < deadband:
            heading = 0.0

        # 시간 EMA 평활
        self.heading_filtered = alpha * heading + (1.0 - alpha) * self.heading_filtered
        return self.heading_filtered


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
