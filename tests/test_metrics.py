"""
test_metrics.py — episode metrics, confidence intervals and the feasibility oracle.
"""

import math

import numpy as np
import pytest

from safenav.evaluation.feasibility import check_feasibility
from safenav.evaluation.metrics import (EpisodeResult, mean_abs_change, path_length, summarize,
                                        t_interval, wilson_interval)


def _result(**kw):
    base = dict(method="m", stage="s", episode_seed=0, safety_filter=True, success=True,
                obstacle_collision=False, out_of_bounds=False, timeout=False, steps=50,
                episode_return=1.0, straight_line=4.0, path_length=5.0, min_clearance=0.4,
                min_wall_clearance=1.0, mean_risk=0.0, max_risk=0.0, interventions=10,
                mean_abs_dv=0.0, mean_abs_domega=0.1, mean_speed=0.8)
    base.update(kw)
    return EpisodeResult(**base)


def test_path_length_and_smoothness():
    xy = np.array([[0, 0], [3, 4], [3, 5]], dtype=float)
    assert path_length(xy) == pytest.approx(6.0)
    assert path_length(xy[:1]) == 0.0
    assert mean_abs_change([0.0, 0.5, -0.5]) == pytest.approx(0.75)
    assert mean_abs_change([1.0]) == 0.0


def test_derived_metrics():
    r = _result()
    assert r.path_efficiency == pytest.approx(0.8)
    assert r.spl == pytest.approx(0.8)
    assert r.intervention_rate == pytest.approx(0.2)
    assert r.time_to_goal == pytest.approx(5.0)
    assert r.outcome == "success"
    failed = _result(success=False, timeout=True)
    assert failed.spl == 0.0 and math.isnan(failed.time_to_goal) and failed.outcome == "timeout"
    assert _result(success=False, out_of_bounds=True).collision


def test_wilson_interval_known_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_t_interval():
    mean, low, high = t_interval([0.8, 0.9, 1.0])
    assert mean == pytest.approx(0.9)
    half = 4.302653 * 0.1 / np.sqrt(3)       # t_{0.975, 2} * s / sqrt(n)
    assert (low, high) == pytest.approx((0.9 - half, 0.9 + half), abs=1e-4)
    assert math.isnan(t_interval([0.5])[1])


def test_summarize_rates():
    results = [_result(), _result(success=False, timeout=True),
               _result(success=False, obstacle_collision=True), _result()]
    s = summarize(results)
    assert s["episodes"] == 4
    assert s["success"]["rate"] == 0.5
    assert s["collision"]["rate"] == 0.25
    assert s["timeout"]["rate"] == 0.25
    assert s["path_efficiency_success"] == pytest.approx(0.8)


def test_feasibility_open_and_blocked():
    free = check_feasibility((2, 5), (8, 5), [], 10.0, 0.3, 0.5)
    assert free.reachable and free.shortest_path == pytest.approx(5.5, abs=0.06)
    wall = [(5.0, y, 0.6) for y in np.arange(0.0, 10.5, 1.0)]
    assert not check_feasibility((2, 5), (8, 5), wall, 10.0, 0.3, 0.5).reachable


def test_feasibility_depends_on_margin():
    # Two obstacles leave a 1.0 m gap; the robot (0.6 m wide) fits only without a 0.3 m margin.
    obstacles = [(5.0, y, 1.0) for y in (3.5, 6.5, 1.5, -0.5, 8.5, 10.5)]
    assert check_feasibility((2, 5), (8, 5), obstacles, 10.0, 0.3, 0.5, margin=0.0).reachable
    assert not check_feasibility((2, 5), (8, 5), obstacles, 10.0, 0.3, 0.5, margin=0.3).reachable
