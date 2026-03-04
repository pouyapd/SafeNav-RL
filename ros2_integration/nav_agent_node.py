"""
nav_agent_node.py

ROS2 node that wraps a trained SafeNav-RL policy for real robot deployment.

This node:
  - Subscribes to odometry (/odom) and goal (/goal_pose)
  - Runs the trained policy at each control tick
  - Publishes velocity commands (/cmd_vel)
  - Applies the safety layer before sending commands

Usage (after sourcing your ROS2 workspace):
    ros2 run safenav_rl nav_agent_node --ros-args -p model_path:=checkpoints/best_model.pt

Prerequisites:
    - ROS2 Humble (https://docs.ros.org/en/humble/Installation.html)
    - pip install rclpy
    - PyTorch and the SafeNav-RL package installed

NOTE: This is a deployment scaffold — it requires a real robot or
      a ROS2 simulator (e.g., Gazebo) with /odom and /goal_pose topics.
"""

# ── ROS2 guard ────────────────────────────────────────────────────────────────
# This guard allows the file to be imported without crashing
# if ROS2 is not installed (e.g., during unit tests).
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import LaserScan
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class for type checking

import numpy as np
import torch
import yaml
import math
from typing import Optional

from models.actor_critic import ActorCritic
from models.safety_layer import SafetyLayer


class NavAgentNode(Node):
    """
    ROS2 navigation agent node.

    Subscribes:
      /odom        (nav_msgs/Odometry)      — robot pose and velocity
      /goal_pose   (geometry_msgs/PoseStamped) — navigation goal
      /scan        (sensor_msgs/LaserScan)  — optional: for safety layer

    Publishes:
      /cmd_vel     (geometry_msgs/Twist)    — velocity command
    """

    def __init__(self):
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS2 not available. Install rclpy and source your ROS2 workspace."
            )
        super().__init__("safenav_rl_agent")

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter("model_path", "checkpoints/best_model.pt")
        self.declare_parameter("config_path", "configs/default_config.yaml")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("deterministic", True)

        model_path = self.get_parameter("model_path").value
        config_path = self.get_parameter("config_path").value
        control_rate = self.get_parameter("control_rate_hz").value
        self.deterministic = self.get_parameter("deterministic").value

        # ── Load Config and Model ─────────────────────────────────
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cpu")  # CPU for inference on robot
        self.model = ActorCritic(self.config).to(self.device)
        self.safety_layer = SafetyLayer(self.config)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.get_logger().info(f"Model loaded from {model_path}")

        # ── State ─────────────────────────────────────────────────
        self.robot_pos = np.zeros(2)
        self.robot_theta = 0.0
        self.robot_vel = np.zeros(2)
        self.goal_pos: Optional[np.ndarray] = None

        # ── Subscribers ───────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_callback, 10
        )

        # ── Publisher ─────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Control Timer ─────────────────────────────────────────
        period = 1.0 / control_rate
        self.timer = self.create_timer(period, self._control_tick)

        self.get_logger().info(
            f"SafeNav-RL agent running at {control_rate} Hz. "
            f"Waiting for /goal_pose..."
        )

    def _odom_callback(self, msg: "Odometry"):
        """Update robot pose from odometry."""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)

    def _goal_callback(self, msg: "PoseStamped"):
        """Update goal position from navigation goal."""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
        ])
        self.get_logger().info(f"New goal received: {self.goal_pos}")

    def _build_observation(self) -> np.ndarray:
        """
        Build the observation vector matching the training environment.
        
        In a real robot deployment, this bridges the sim-to-real gap
        by normalizing the same features as the training environment.
        """
        workspace_size = self.config["env"]["workspace_size"]
        safe_distance = self.config["safety"]["safe_distance"]

        x_norm = self.robot_pos[0] / workspace_size
        y_norm = self.robot_pos[1] / workspace_size
        sin_theta = math.sin(self.robot_theta)
        cos_theta = math.cos(self.robot_theta)

        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        max_dist = workspace_size * math.sqrt(2)
        d_goal_norm = math.sqrt(dx ** 2 + dy ** 2) / max_dist

        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.robot_theta
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        angle_error_norm = angle_error / math.pi

        # In real deployment: use LaserScan to compute risk score
        # For this scaffold, we set risk to 0 (no obstacle info integrated yet)
        risk = 0.0

        return np.array(
            [x_norm, y_norm, sin_theta, cos_theta, d_goal_norm, angle_error_norm, risk],
            dtype=np.float32,
        )

    def _control_tick(self):
        """Main control loop — runs at control_rate_hz."""
        if self.goal_pos is None:
            return

        # Check if goal reached
        dist = np.linalg.norm(self.goal_pos - self.robot_pos)
        if dist < self.config["env"]["goal_radius"]:
            self._stop_robot()
            self.get_logger().info("Goal reached!")
            self.goal_pos = None
            return

        # Build observation and run policy
        obs = self._build_observation()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self.deterministic:
                action = self.model.get_deterministic_action(obs_tensor).squeeze(0)
            else:
                action, _, _ = self.model(obs_tensor)
                action = action.squeeze(0)

        action_np = action.cpu().numpy()

        # Apply safety layer (no obstacle info from scan yet — TODO: integrate LaserScan)
        safe_action = self.safety_layer.project_action(
            action_np, self.robot_pos, self.robot_theta, obstacles=[]
        )

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = float(safe_action[0])
        cmd.angular.z = float(safe_action[1])
        self.cmd_vel_pub.publish(cmd)

    def _stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)


def main():
    if not ROS2_AVAILABLE:
        print("ERROR: ROS2 not available. Source your ROS2 installation first.")
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
