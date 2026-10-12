# ROS 2 integration (untested scaffold)

This folder contains a ROS 2 node that wraps a trained SafeNav-RL policy. **It has
not been run on a robot or in a simulator such as Gazebo.** It is a starting point
for a deployment, not evidence that the policy works on hardware.

## What is in the folder

| File | Content | Tested |
|---|---|---|
| `nav_agent_node.py` | ROS 2 node: `/odom`, `/goal_pose`, `/scan` in, `/cmd_vel` out | imports without ROS; never run with ROS |
| `scan_utils.py` | converts a LaserScan into the policy's 16 range beams and into point obstacles for the safety filter | yes, unit tests in `tests/test_ros2_scan_utils.py` |
| `topic_definitions.py` | topic names, frames and velocity limits | — |

## Running (after installing ROS 2 Humble and sourcing it)

```bash
# from the repository root, with the package installed (pip install -e .)
python ros2_integration/nav_agent_node.py --ros-args \
  -p model_path:=runs/default/checkpoints/best.pt \
  -p control_rate_hz:=10.0

# send a goal
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'odom'}, pose: {position: {x: 5.0, y: 5.0, z: 0.0}}}"
```

The node reads the training configuration from the checkpoint, so the beam count,
range limit and safety margin match the trained agent.

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/odom` | nav_msgs/Odometry | subscribe |
| `/goal_pose` | geometry_msgs/PoseStamped | subscribe |
| `/scan` | sensor_msgs/LaserScan | subscribe |
| `/cmd_vel` | geometry_msgs/Twist | publish |

## Known gaps before any real use

1. **No safety guarantee on a robot.** The simulator's guarantee (see
   [`docs/SAFETY.md`](../docs/SAFETY.md)) needs exact, complete obstacle geometry
   and the exact unicycle model. The node uses scan returns as point obstacles,
   which miss anything between rays and ignore delay, slip and odometry drift.
2. **Frames.** The observation uses odometry coordinates normalised by the
   training workspace size (10 m). There is no map frame, TF lookup or
   relocalisation.
3. **Sim-to-real gap.** The policy was trained only in the 2D simulator with
   circular static obstacles and idealised kinematics; real sensor noise, moving
   people and robot dynamics were never modelled.
4. **Velocity limits.** Training used v in [0, 1] m/s and omega in [-1, 1] rad/s;
   adjust `topic_definitions.py` and retrain for another platform.

Suggested next steps: test in Gazebo with a TurtleBot3 model, add obstacle
extraction (circle fitting) from scans, and measure the closed-loop clearance.
