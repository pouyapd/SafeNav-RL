"""
test_ros2_scan_utils.py — LaserScan conversion helpers (no ROS installation needed).
"""

import numpy as np
import pytest

from ros2_integration import nav_agent_node
from ros2_integration.scan_utils import risk_from_scan, scan_to_beams, scan_to_point_obstacles


def test_node_module_imports_without_ros():
    assert nav_agent_node.ROS2_AVAILABLE in (True, False)


def test_scan_to_beams_takes_minimum_per_sector():
    n = 360
    ranges = np.full(n, 5.0)
    ranges[180] = 1.0                       # angle_min=-pi -> index 180 is straight ahead
    ranges[0] = np.inf                      # no return behind the robot
    offsets = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    beams = scan_to_beams(ranges, -np.pi, 2 * np.pi / n, offsets, max_range=3.0)
    ahead = int(np.argmin(np.abs(offsets)))
    assert beams[ahead] == pytest.approx(1.0)
    assert np.all(np.delete(beams, ahead) == 3.0)


def test_point_obstacles_in_odom_frame():
    ranges = [np.nan, 2.0, 9.0]
    obs = scan_to_point_obstacles(ranges, -np.pi / 2, np.pi / 2, robot_xy=(1.0, 1.0), robot_yaw=np.pi / 2,
                                  max_range=3.0)
    assert len(obs) == 1                    # NaN and out-of-range returns are dropped
    assert (obs[0].x, obs[0].y, obs[0].radius) == pytest.approx((1.0, 3.0, 0.0))


def test_risk_from_scan():
    assert risk_from_scan([0.4, 2.0], risk_distance=0.8, max_range=3.0) == pytest.approx(0.5)
    assert risk_from_scan([np.inf], risk_distance=0.8, max_range=3.0) == 0.0
