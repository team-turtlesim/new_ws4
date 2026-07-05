"""Lane-following lateral controller.

Subscribes LaneInfo on ``/lane_info``, runs a PD controller on the lateral
offset (plus an optional heading term) to keep the car in the lane centre, and
publishes the resulting steering/throttle as a ``Control`` message on
``/control``.

This node does NOT touch the actuator hardware. ``control_node`` remains the
single owner of the PCA9685 and consumes ``/control``. Keeping the command bus
means the monitor dashboard can display the steering/throttle we generate, which
is essential for validating the steering polarity/gain before the car ever
moves.

Safety-first design (see racer priorities):
  * Fixed-rate publish loop, independent of when LaneInfo arrives.
  * Watchdog: if no LaneInfo for ``lost_timeout_sec`` -> neutral steer + stop.
  * Low confidence (lane lost) -> throttle ramps to zero.
  * Throttle slew-rate limited to avoid jerks.
  * ``cruise_throttle`` defaults to 0.0: on first run the node computes and
    publishes steering but keeps the car still, so the steering sign/gain can be
    validated on the dashboard. Raise it only after that check.
"""

import os
from pathlib import Path

import rclpy
from rclpy.node import Node
import yaml

from interface.msg import Control, LaneInfo


def get_default_vehicle_config_path():
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/D-Racer/src/config/vehicle_config.yaml'


def clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


class LaneFollowNode(Node):
    def __init__(self):
        super().__init__('lane_follow_node')

        # --- Topics / IO ---
        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('lane_info_topic', '/lane_info')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('control_hz', 20.0)

        # --- Steering (PD on lateral offset + optional heading term) ---
        # logical steer: positive = steer physically RIGHT (matches LaneInfo:
        # +offset means lane centre is right of image -> car drifted left ->
        # steer right). steer_sign maps that to the servo percent polarity;
        # -1.0 is the carry-over value from the previous car, VERIFY on track.
        self.declare_parameter('kp_offset', 0.6)
        self.declare_parameter('kd_offset', 0.15)
        # Integral term: removes steady-state offset so the car reaches the true
        # centre on straights (P alone leaves residual error under a constant
        # bias like camera/trim offset). Clamped + reset on stop for anti-windup.
        self.declare_parameter('ki_offset', 0.4)
        self.declare_parameter('i_limit', 0.3)     # clamp on |ki*integral|
        self.declare_parameter('k_heading', 0.0)   # heading proxy noisy; start 0
        self.declare_parameter('steer_limit', 0.7)  # max |logical steer|
        self.declare_parameter('steer_sign', -1.0)
        # steer_trim (straight/neutral steering value) comes from vehicle_config.
        # --- Smoothing (kill twitch on straights) ---
        # d_offset spikes when detection briefly drops and offset jumps on
        # recovery -> clamp it. steer EMA low-passes the final command.
        self.declare_parameter('d_offset_limit', 2.0)     # clamp |d(offset)/dt|
        self.declare_parameter('steer_smooth_alpha', 0.4)  # 1.0 = no smoothing
        # --- Curve bias (aim off-centre through curves) ---
        # target offset = curve_bias * heading_error. On a right curve
        # (heading_error>0) this makes a positive target -> car hugs left,
        # correcting the tendency to run wide out the right lane.
        self.declare_parameter('curve_bias', 0.0)
        self.declare_parameter('offset_target_limit', 0.4)  # clamp the bias

        # --- Throttle ---
        self.declare_parameter('cruise_throttle', 0.0)  # 0 => validate steer first
        self.declare_parameter('max_throttle', 0.30)
        self.declare_parameter('throttle_slew_per_sec', 0.6)  # ramp rate

        # --- Fail-safes ---
        self.declare_parameter('min_confidence', 0.2)   # below -> treat as lost
        self.declare_parameter('lost_timeout_sec', 0.5)  # no msg -> stop
        self.declare_parameter('debug_log', False)

        self.lane_info_topic = str(self.get_parameter('lane_info_topic').value)
        self.control_topic = str(self.get_parameter('control_topic').value)
        control_hz = float(self.get_parameter('control_hz').value)
        if control_hz <= 0.0:
            raise ValueError('control_hz must be greater than 0')
        self.control_hz = control_hz

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        self.steer_trim = self.load_steer_trim()

        # runtime state
        self.last_offset = 0.0
        self.last_msg = None            # latest LaneInfo
        self.last_msg_time = None       # rclpy Time of last LaneInfo
        self.prev_offset_for_d = 0.0
        self.prev_ctrl_time = None      # for derivative + slew dt
        self.throttle_cmd = 0.0
        self.steer_cmd_filtered = self.steer_trim  # EMA-smoothed steering out
        self.was_low_conf = False       # prev frame confidence below min?
        self.integral = 0.0             # integral of offset error (I term)

        self.pub = self.create_publisher(Control, self.control_topic, 10)
        self.create_subscription(
            LaneInfo,
            self.lane_info_topic,
            self.lane_info_callback,
            10,
        )
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)

        self.get_logger().info(
            'lane_follow_node started:\n'
            f'  lane_info_topic={self.lane_info_topic}\n'
            f'  control_topic={self.control_topic}\n'
            f'  control_hz={self.control_hz}\n'
            f'  steer_trim={self.steer_trim}\n'
            f'  steer_sign={float(self.get_parameter("steer_sign").value)}\n'
            f'  kp={float(self.get_parameter("kp_offset").value)} '
            f'kd={float(self.get_parameter("kd_offset").value)} '
            f'kh={float(self.get_parameter("k_heading").value)}\n'
            f'  cruise_throttle={float(self.get_parameter("cruise_throttle").value)} '
            f'(0 => steering-only; raise after validating sign/gain)\n'
            f'  max_throttle={float(self.get_parameter("max_throttle").value)}'
        )

    def lane_info_callback(self, msg: LaneInfo):
        self.last_msg = msg
        self.last_msg_time = self.get_clock().now()

    def control_loop(self):
        now = self.get_clock().now()

        # dt for derivative / slew, robust to first tick and clock jumps.
        if self.prev_ctrl_time is None:
            dt = 1.0 / self.control_hz
        else:
            dt = (now - self.prev_ctrl_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / self.control_hz
        self.prev_ctrl_time = now

        # --- Watchdog: no lane info at all -> hold straight, stop. ---
        if self.last_msg is None or self.last_msg_time is None:
            self.publish_stop('waiting for lane_info')
            return
        age = (now - self.last_msg_time).nanoseconds * 1e-9
        lost_timeout = float(self.get_parameter('lost_timeout_sec').value)
        if age > lost_timeout:
            self.throttle_cmd = 0.0
            self.publish_stop(f'lane_info stale ({age:.2f}s > {lost_timeout:.2f}s)')
            return

        msg = self.last_msg
        offset = float(msg.lane_offset)
        heading = float(msg.heading_error)
        confidence = float(msg.confidence)

        kp = float(self.get_parameter('kp_offset').value)
        kd = float(self.get_parameter('kd_offset').value)
        ki = float(self.get_parameter('ki_offset').value)
        i_limit = float(self.get_parameter('i_limit').value)
        kh = float(self.get_parameter('k_heading').value)
        steer_limit = float(self.get_parameter('steer_limit').value)
        steer_sign = float(self.get_parameter('steer_sign').value)
        d_limit = float(self.get_parameter('d_offset_limit').value)
        alpha = clip(float(self.get_parameter('steer_smooth_alpha').value), 0.05, 1.0)
        curve_bias = float(self.get_parameter('curve_bias').value)
        target_limit = float(self.get_parameter('offset_target_limit').value)
        min_conf = float(self.get_parameter('min_confidence').value)

        low_conf = confidence < min_conf

        # --- Curve bias: aim off-centre through curves. ---
        # right curve (heading>0) -> positive target -> hug left.
        offset_target = clip(curve_bias * heading, -target_limit, target_limit)
        error = offset - offset_target

        # --- Derivative, clamped; reset on detection recovery to avoid slam. ---
        if self.was_low_conf and not low_conf:
            # just re-acquired the lane: offset jumped from a stale HOLD value,
            # so the raw derivative is garbage this frame -> suppress it.
            self.prev_offset_for_d = offset
        d_offset = clip((offset - self.prev_offset_for_d) / dt, -d_limit, d_limit)
        self.prev_offset_for_d = offset
        self.was_low_conf = low_conf

        # --- Integral (anti-windup): only accumulate when tracking & moving;
        #     freeze/reset otherwise. i_term clamped to avoid runaway. ---
        if low_conf or self.throttle_cmd <= 0.0:
            self.integral = 0.0
        elif ki > 0.0:
            self.integral += error * dt
            self.integral = clip(self.integral, -i_limit / ki, i_limit / ki)
        i_term = clip(ki * self.integral, -i_limit, i_limit)

        # --- PID(+heading) then EMA low-pass for smooth output. ---
        logical = kp * error + i_term + kd * d_offset + kh * heading
        logical = clip(logical, -steer_limit, steer_limit)
        steering_raw = clip(self.steer_trim + steer_sign * logical, -1.0, 1.0)
        self.steer_cmd_filtered = (
            alpha * steering_raw + (1.0 - alpha) * self.steer_cmd_filtered
        )
        steering = clip(self.steer_cmd_filtered, -1.0, 1.0)

        # --- Throttle: cruise unless lane lost; slew-limited. ---
        cruise = float(self.get_parameter('cruise_throttle').value)
        max_throttle = float(self.get_parameter('max_throttle').value)
        slew = float(self.get_parameter('throttle_slew_per_sec').value)

        target_throttle = 0.0 if low_conf else clip(cruise, 0.0, max_throttle)

        step = slew * dt
        if target_throttle > self.throttle_cmd:
            self.throttle_cmd = min(self.throttle_cmd + step, target_throttle)
        else:
            self.throttle_cmd = max(self.throttle_cmd - step, target_throttle)

        self.publish_control(steering, self.throttle_cmd)

        if bool(self.get_parameter('debug_log').value):
            self.get_logger().info(
                f'off={offset:+.3f} tgt={offset_target:+.3f} i={i_term:+.3f} '
                f'd={d_offset:+.3f} hd={heading:+.3f} conf={confidence:.2f} '
                f'-> steer={steering:+.3f} thr={self.throttle_cmd:.3f}'
            )

    def publish_control(self, steering, throttle):
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        self.pub.publish(msg)

    def publish_stop(self, reason=''):
        # Neutral steering, zero throttle. Throttle already forced to 0 by caller
        # for the stale case; ramp it down here otherwise for a soft stop.
        self.steer_cmd_filtered = self.steer_trim  # restart EMA from neutral
        self.integral = 0.0                        # anti-windup: no drift while stopped
        self.publish_control(self.steer_trim, 0.0)
        if bool(self.get_parameter('debug_log').value) and reason:
            self.get_logger().warning(f'STOP: {reason}', throttle_duration_sec=1.0)

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

    def destroy_node(self):
        try:
            # Best-effort stop on shutdown.
            self.publish_control(self.steer_trim, 0.0)
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
