"""
scan_utils.py — ROS-independent helpers used by nav_agent_node.py.

They convert a sensor_msgs/LaserScan (given as plain arrays) into
  * the range-beam layout the policy was trained with, and
  * point obstacles for the CBF safety filter.
Being plain NumPy, they are unit-tested without a ROS installation
(tests/test_ros2_scan_utils.py).

Limitation: point obstacles from a scan only cover the directions the scanner
sampled, so the filter's invariance argument (exact, complete obstacle
knowledge) does not hold on a real robot. This is a deployment scaffold that
has not been tested on hardware or in Gazebo.
"""

from typing import List, Sequence

import numpy as np

from safenav.env.obstacle_map import Obstacle


def _clean(ranges: Sequence[float], max_range: float) -> np.ndarray:
    r = np.asarray(ranges, dtype=float)
    r = np.where(np.isfinite(r) & (r > 0), r, max_range)   # inf/NaN/0 = no return
    return np.clip(r, 0.0, max_range)


def scan_angles(n: int, angle_min: float, angle_increment: float) -> np.ndarray:
    """Beam angles of a LaserScan in the sensor frame."""
    return angle_min + angle_increment * np.arange(n)


def scan_to_beams(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    beam_offsets: Sequence[float],
    max_range: float,
) -> np.ndarray:
    """
    Resample a scan to the training beam layout: for every beam offset (radians,
    relative to the robot heading) take the minimum range of the scan rays within
    half a beam spacing. Beams with no scan ray return max_range.
    """
    r = _clean(ranges, max_range)
    angles = scan_angles(len(r), angle_min, angle_increment)
    offsets = np.asarray(beam_offsets, dtype=float)
    half = np.pi / max(len(offsets), 1)
    out = np.full(len(offsets), float(max_range))
    for k, off in enumerate(offsets):
        diff = np.abs((angles - off + np.pi) % (2 * np.pi) - np.pi)
        sel = diff <= half
        if np.any(sel):
            out[k] = r[sel].min()
    return out


def scan_to_point_obstacles(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    robot_xy: Sequence[float],
    robot_yaw: float,
    max_range: float,
    stride: int = 1,
) -> List[Obstacle]:
    """Scan returns (closer than max_range) as zero-radius obstacles in the odometry frame."""
    r = _clean(ranges, max_range)
    angles = scan_angles(len(r), angle_min, angle_increment)
    keep = np.arange(len(r))[::max(1, stride)]
    keep = keep[r[keep] < max_range]
    world = robot_yaw + angles[keep]
    xs = robot_xy[0] + r[keep] * np.cos(world)
    ys = robot_xy[1] + r[keep] * np.sin(world)
    return [Obstacle(float(x), float(y), 0.0) for x, y in zip(xs, ys)]


def risk_from_scan(ranges: Sequence[float], risk_distance: float, max_range: float) -> float:
    """Same definition as the simulator: 1 at an obstacle surface, 0 at risk_distance."""
    r = _clean(ranges, max_range)
    if r.size == 0 or risk_distance <= 0:
        return 0.0
    return float(np.clip(1.0 - r.min() / risk_distance, 0.0, 1.0))
