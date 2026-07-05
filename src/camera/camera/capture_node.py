"""데이터셋 수집 노드.

카메라 압축영상(/camera/image/compressed)을 구독해 학습용 이미지를 디스크에
저장한다. 이미 JPEG 로 들어오므로 재인코딩 없이 그대로 써서 CPU 부하가 거의 없다.

주행 스택(racer_bringup)과 별개로, 데이터 수집할 때만 따로 실행한다:
    ros2 run camera capture_node
    ros2 run camera capture_node --ros-args -p save_every_n:=20 -p save_dir:=/media/topst/USB

라이브 제어(파라미터는 매 프레임 다시 읽음):
    ros2 param set /capture_node enabled false   # 잠깐 멈춤
    ros2 param set /capture_node enabled true     # 다시 시작
"""

import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class CaptureNode(Node):
    def __init__(self):
        super().__init__('capture_node')

        self.declare_parameter('subscribe_topic', '/camera/image/compressed')
        # 보드 내장 eMMC. (df 상 20GB 여유 — 640x480 JPEG ~60KB 라 수십 시간 분량 저장 가능)
        self.declare_parameter('save_dir', '~/dataset')
        # N 프레임마다 1장 저장. 카메라 20fps 기준 10 -> 약 2fps.
        # 근접 중복 프레임을 줄여 라벨링 부담과 데이터 편향을 낮춘다.
        self.declare_parameter('save_every_n', 10)
        # 라이브로 수집을 켜고 끌 수 있는 스위치(매 프레임 다시 읽음).
        self.declare_parameter('enabled', True)
        self.declare_parameter('debug_log', True)

        subscribe_topic = str(self.get_parameter('subscribe_topic').value)
        self.save_dir = os.path.expanduser(str(self.get_parameter('save_dir').value))
        self.save_every_n = max(1, int(self.get_parameter('save_every_n').value))
        self.debug_log = bool(self.get_parameter('debug_log').value)

        os.makedirs(self.save_dir, exist_ok=True)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            CompressedImage,
            subscribe_topic,
            self.image_callback,
            image_qos,
        )

        self.frame_count = 0
        self.saved_count = 0

        self.get_logger().info(
            'capture node started (dataset collection):\n'
            f'  subscribe_topic={subscribe_topic}\n'
            f'  save_dir={self.save_dir}\n'
            f'  save_every_n={self.save_every_n} (save 1 of every N frames)\n'
            f'  enabled={bool(self.get_parameter("enabled").value)}'
        )

    def image_callback(self, msg: CompressedImage):
        if not bool(self.get_parameter('enabled').value):
            return

        self.frame_count += 1
        if self.frame_count % self.save_every_n != 0:
            return

        # 밀리초까지 넣어 파일명 충돌 방지.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = os.path.join(self.save_dir, f'frame_{stamp}.jpg')

        try:
            # 이미 JPEG 이므로 재인코딩 없이 바이트 그대로 기록.
            with open(filename, 'wb') as out_file:
                out_file.write(bytes(msg.data))
        except OSError as exc:
            self.get_logger().warning(f'Failed to save {filename}: {exc}')
            return

        self.saved_count += 1
        if self.debug_log:
            self.get_logger().info(f'Saved {filename} (total {self.saved_count})')


def main(args=None):
    rclpy.init(args=args)
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
