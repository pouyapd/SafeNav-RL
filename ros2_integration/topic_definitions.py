"""
topic_definitions.py

Defines all ROS2 topics and message types used by SafeNav-RL.

Centralizing topic names here prevents typos and makes
remapping easier when deploying on different robot platforms.
"""

# ── Topic Names ───────────────────────────────────────────────────────────────
TOPIC_ODOM = "/odom"                  # nav_msgs/Odometry
TOPIC_GOAL = "/goal_pose"             # geometry_msgs/PoseStamped
TOPIC_LASER = "/scan"                 # sensor_msgs/LaserScan
TOPIC_CMD_VEL = "/cmd_vel"            # geometry_msgs/Twist

# ── QoS Profiles ─────────────────────────────────────────────────────────────
# For real-time control, use BEST_EFFORT with small queue sizes
CONTROL_QOS_DEPTH = 10
SENSOR_QOS_DEPTH = 5

# ── Frame IDs ─────────────────────────────────────────────────────────────────
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
MAP_FRAME = "map"

# ── Control ───────────────────────────────────────────────────────────────────
DEFAULT_CONTROL_RATE_HZ = 10.0
MAX_LINEAR_VEL = 1.0     # m/s — matches training environment
MAX_ANGULAR_VEL = 1.0    # rad/s — matches training environment

# ── Message Type Strings (for documentation) ─────────────────────────────────
MSG_TYPES = {
    TOPIC_ODOM: "nav_msgs/msg/Odometry",
    TOPIC_GOAL: "geometry_msgs/msg/PoseStamped",
    TOPIC_LASER: "sensor_msgs/msg/LaserScan",
    TOPIC_CMD_VEL: "geometry_msgs/msg/Twist",
}
