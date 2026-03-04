# ROS2 Integration Guide

This directory contains the ROS2 deployment scaffold for the trained SafeNav-RL policy.

## Prerequisites

1. **ROS2 Humble** installed: https://docs.ros.org/en/humble/Installation.html
2. Source your ROS2 installation:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
3. Install Python dependencies:
   ```bash
   pip install rclpy torch pyyaml
   ```

## Running the Agent

```bash
# Deploy with default settings
ros2 run safenav_rl nav_agent_node \
  --ros-args \
  -p model_path:=checkpoints/best_model.pt \
  -p config_path:=configs/default_config.yaml \
  -p control_rate_hz:=10.0

# Remap topics for a specific robot (e.g., TurtleBot3)
ros2 run safenav_rl nav_agent_node \
  --ros-args \
  -p model_path:=checkpoints/best_model.pt \
  --remap /odom:=/tb3/odom \
  --remap /cmd_vel:=/tb3/cmd_vel
```

## Sending a Goal

```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 5.0, z: 0.0}}}"
```

## Topic Overview

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/odom` | nav_msgs/Odometry | Subscribe | Robot pose from wheel odometry |
| `/goal_pose` | geometry_msgs/PoseStamped | Subscribe | Navigation goal |
| `/scan` | sensor_msgs/LaserScan | Subscribe | (TODO) Obstacle detection |
| `/cmd_vel` | geometry_msgs/Twist | Publish | Velocity commands |

## Sim-to-Real Notes

When deploying from simulation to a real robot, consider:

1. **Observation normalization**: The workspace size in `default_config.yaml` must match your real environment dimensions.
2. **Velocity limits**: Adjust `MAX_LINEAR_VEL` and `MAX_ANGULAR_VEL` in `topic_definitions.py` to your robot's physical limits.
3. **Control rate**: Start with 5–10 Hz for safety; increase after validation.
4. **LaserScan integration**: The current scaffold does not yet feed laser scan data into the safety layer. This is the recommended next step for real deployment.

## TODO for Full Real-Robot Deployment

- [ ] Integrate `/scan` LaserScan into safety layer obstacle detection
- [ ] Add TF2 transform listener for accurate localization
- [ ] Test in Gazebo simulation before real hardware
- [ ] Add ROS2 action server interface for waypoint following
