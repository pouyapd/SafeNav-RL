"""
nav_agent_node.py

ROS 2 node that wraps a trained SafeNav-RL policy. UNTESTED SCAFFOLD: it has not
been run on a robot or in a simulator such as Gazebo.

This node:
  - Subscribes to odometry (/odom), a goal (/goal_pose) and a laser scan (/scan)
  - Builds the same observation vector as the training environment
    (range beams are resampled from the scan, see scan_utils.py)
  - Runs the policy at each control tick
  - Applies the CBF speed filter with scan returns as point obstacles
  - Publishes velocity commands (/cmd_vel)

Usage (after sourcing your ROS 2 workspace, from the repository root):
    python ros2_integration/nav_agent_node.py --ros-args -p model_path:=runs/default/checkpoints/best.pt

Known gaps before any real use (see README.md in this folder):
    - observation uses odometry coordinates normalised by the training
      workspace size; a real map frame and TF are not handled
    - the filter's guarantee needs exact, complete obstacle geometry; a scan
      gives neither, so no safety guarantee carries over to a real robot
"""

# ── ROS2 guard ────────────────────────────────────────────────────────────────
# Allows importing this file without ROS 2 (e.g. during unit tests).
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class

import math
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ros2_integration.scan_utils import risk_from_scan, scan_to_beams, scan_to_point_obstacles  # noqa: E402
from safenav.agents.policy import PPOPolicy, load_checkpoint  # noqa: E402
from safenav.safety.cbf_filter import CBFSafetyFilter  # noqa: E402


class NavAgentNode(Node):
    """
    ROS 2 navigation agent node.

    Subscribes:
      /odom        (nav_msgs/Odometry)         — robot pose
      /goal_pose   (geometry_msgs/PoseStamped) — navigation goal
      /scan        (sensor_msgs/LaserScan)     — range sensor

    Publishes:
      /cmd_vel     (geometry_msgs/Twist)       — velocity command
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 not available. Install rclpy and source your ROS 2 workspace.")
        super().__init__("safenav_rl_agent")

        self.declare_parameter("model_path", "runs/default/checkpoints/best.pt")
        self.declare_parameter("control_rate_hz", 10.0)
        model_path = self.get_parameter("model_path").value
        control_rate = self.get_parameter("control_rate_hz").value

        # The checkpoint stores the configuration it was trained with.
        self.config = load_checkpoint(model_path)["config"]
        self.policy = PPOPolicy.from_checkpoint(model_path, deterministic=True)
        self.safety_filter = CBFSafetyFilter.from_config(self.config)
        self.safety_filter.workspace_size = None   # no training-workspace walls on a real robot
        self.get_logger().info(f"Model loaded from {model_path}")

        sensor = self.config.get("sensor", {}) or {}
        self.num_beams = int(sensor.get("num_beams", 0))
        self.max_range = float(sensor.get("max_range", 3.0))
        self.beam_offsets = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)

        self.robot_pos = np.zeros(2)
        self.robot_theta = 0.0
        self.goal_pos: Optional[np.ndarray] = None
        self.scan: Optional[LaserScan] = None

        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_callback, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_callback, 5)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(1.0 / control_rate, self._control_tick)
        self.get_logger().info(f"SafeNav-RL agent running at {control_rate} Hz. Waiting for /goal_pose and /scan...")

    def _odom_callback(self, msg: "Odometry"):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    def _goal_callback(self, msg: "PoseStamped"):
        self.goal_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f"New goal received: {self.goal_pos}")

    def _scan_callback(self, msg: "LaserScan"):
        self.scan = msg

    def _build_observation(self) -> np.ndarray:
        """Same layout as NavigationEnv._get_obs (see safenav/env/navigation_env.py)."""
        env_cfg = self.config["env"]
        workspace_size = env_cfg["workspace_size"]
        risk_distance = env_cfg.get("risk_distance", self.config["safety"]["safe_distance"])

        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        angle_error = (math.atan2(dy, dx) - self.robot_theta + math.pi) % (2 * math.pi) - math.pi
        ranges = self.scan.ranges
        base = [
            self.robot_pos[0] / workspace_size,
            self.robot_pos[1] / workspace_size,
            math.sin(self.robot_theta),
            math.cos(self.robot_theta),
            math.hypot(dx, dy) / (workspace_size * math.sqrt(2)),
            angle_error / math.pi,
            risk_from_scan(ranges, risk_distance, self.max_range),
        ]
        beams = scan_to_beams(ranges, self.scan.angle_min, self.scan.angle_increment,
                              self.beam_offsets, self.max_range) / self.max_range
        obs = np.concatenate([base, beams]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _control_tick(self):
        if self.goal_pos is None or self.scan is None:
            return
        if np.linalg.norm(self.goal_pos - self.robot_pos) < self.config["env"]["goal_radius"]:
            self._stop_robot()
            self.get_logger().info("Goal reached.")
            self.goal_pos = None
            return

        action = self.policy.act(self._build_observation())
        obstacles = scan_to_point_obstacles(self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
                                            self.robot_pos, self.robot_theta, self.max_range, stride=2)
        safe_action, _ = self.safety_filter.filter(action, self.robot_pos, self.robot_theta, obstacles)

        cmd = Twist()
        cmd.linear.x = float(safe_action[0])
        cmd.angular.z = float(safe_action[1])
        self.cmd_vel_pub.publish(cmd)

    def _stop_robot(self):
        self.cmd_vel_pub.publish(Twist())


def main():
    if not ROS2_AVAILABLE:
        print("ERROR: ROS 2 not available. Source your ROS 2 installation first.")
        return
    rclpy.init()
    node = NavAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
