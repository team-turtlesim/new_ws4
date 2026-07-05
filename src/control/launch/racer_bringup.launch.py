"""One-shot bringup for the lane-following racer.

Default (safe) launch starts the perception + web-dashboard stack only:
    camera -> opencv(edge) -> lane_detection(/lane/detection) -> interpret -> /lane_info
    battery, monitor(web dashboard on :5000)
The monitor "edge" pane is pointed at the lane-detection debug overlay so the
dashboard shows ROI line + fitted lanes + centre.

Add `drive:=true` to ALSO start the lateral controller (lane_follow_node) and
the actuator driver (control_node). Even then the car does NOT move until
cruise_throttle is raised (defaults to 0.0) — validate steering first, then:
    ros2 param set /lane_follow_node cruise_throttle 0.17

Usage:
    ros2 launch control racer_bringup.launch.py                 # web + perception
    ros2 launch control racer_bringup.launch.py drive:=true     # + steering/actuator
    ros2 launch control racer_bringup.launch.py debug_overlay:=false  # raw edge pane
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def get_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


# Tuned lateral-control gains from the 2026-07-04 live-drive session.
# 2026-07-05 튜닝: heading_error 가 +0.45 에 포화(단일차선 heading 추정 버그)라
# curve_bias/k_heading 이 직진에서 좌측편향 + 뱀주행을 유발 -> 둘 다 0 으로 분리.
# 순수 offset PID + 감쇠/평활 강화로 직진 안정화(뱀주행 크게 감소, 실주행 확인).
LANE_FOLLOW_PARAMS = {
    'debug_log': True,
    'kp_offset': 0.45,       # 0.55 -> 0.45 과반응 완화
    'kd_offset': 0.12,       # 0.05 -> 0.12 감쇠 강화(뱀주행 억제)
    'ki_offset': 0.2,        # 0.4 -> 0.2 적분 와인드업 흔들림 감소
    'k_heading': 0.0,        # 포화 heading 조향항 제거(코드 주석 권고: noisy -> 0)
    'steer_smooth_alpha': 0.30,  # 0.35 -> 0.30 출력 평활 강화
    'd_offset_limit': 2.0,
    # 단일차선 heading 을 lane_detection 에서 0 처리(heading_require_both_lanes)한 뒤
    # 재활성화. 양쪽 차선 곡선에서 코너 예측 회복. 단일차선 heading garbage 가 없어
    # 직진 뱀주행은 안 생긴다. 곡선 크로싱 심하면 0.4 로, 뱀주행 재발하면 0.2 로.
    'curve_bias': 0.3,
    'cruise_throttle': 0.0,  # SAFE: no motion until raised via param
}


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    cfg = {'vehicle_config_file': vehicle_config_path}

    drive = LaunchConfiguration('drive')
    debug_overlay = LaunchConfiguration('debug_overlay')

    # monitor "edge" pane topic: lane overlay when debug_overlay, else raw edge.
    edge_topic = PythonExpression([
        "'/lane_detection/image/debug' if '", debug_overlay,
        "' == 'true' else '/opencv/image/edge'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'drive', default_value='false',
            description='Also start lane_follow_node + control_node (actuator).',
        ),
        DeclareLaunchArgument(
            'debug_overlay', default_value='true',
            description='Point the monitor edge pane at the lane debug overlay.',
        ),

        # --- Perception + web (always) ---
        Node(package='camera', executable='camera_node', name='camera_node',
             output='screen', parameters=[cfg]),
        Node(package='opencv', executable='opencv_node', name='opencv_node',
             output='screen'),
        Node(package='lane_detection', executable='lane_node',
             name='lane_detection_node', output='screen', parameters=[cfg]),
        # interpret: LaneDetection(인지) -> 시간필터/판단 -> LaneInfo(제어용)
        Node(package='interpret', executable='interpret_node',
             name='interpret_node', output='screen'),
        Node(package='battery', executable='battery_node', name='battery_node',
             output='screen'),
        Node(package='monitor', executable='monitor_node', name='monitor_node',
             output='screen',
             parameters=[cfg, {'opencv_edge_topic': edge_topic}]),

        # --- Steering + actuator (only with drive:=true) ---
        Node(package='control', executable='lane_follow_node',
             name='lane_follow_node', output='screen',
             parameters=[cfg, LANE_FOLLOW_PARAMS],
             condition=IfCondition(drive)),
        Node(package='control', executable='control_node', name='control_node',
             output='screen', parameters=[cfg],
             condition=IfCondition(drive)),
    ])
