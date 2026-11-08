"""Tests for research/experiments/tools/seed_level_stats.py (supplementary seed-level statistics)."""

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_level_stats",
                                              ROOT / "research" / "experiments" / "tools" / "seed_level_stats.py")
sls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sls)


def _episodes(arm, rates, n=10):
    rows = []
    for seed, r in enumerate(rates):
        k = round(r * n)
        for i in range(n):
            outcome = "success" if i < k else "timeout"
            rows.append({"arm": arm, "train_seed": seed, "episode_seed": i, "outcome": outcome,
                         "class": "" if outcome == "success" else "self-stop", "set": "fresh"})
    return rows


def test_per_seed_rates_and_t_interval():
    df = pd.DataFrame(_episodes("a", [0.2, 0.4, 0.6]))
    s = sls.per_seed(df, "success")
    assert list(s.round(6)) == [0.2, 0.4, 0.6]
    d = sls.describe(s)
    assert d["mean"] == pytest.approx(0.4)
    assert d["sd"] == pytest.approx(0.2)
    half = 4.302652729911275 * 0.2 / math.sqrt(3)  # t(0.975, 2)
    assert d["t_ci95"] == pytest.approx([0.4 - half, 0.4 + half])


def test_hedges_g_sign_and_size():
    a = pd.Series([0.5, 0.6, 0.7])
    b = pd.Series([0.1, 0.2, 0.3])
    g = sls.hedges_g(a, b)
    # pooled SD 0.1, Cohen's d 4.0, small-sample factor 1 - 3/(4*6-9) = 0.8
    assert g == pytest.approx(3.2)
    assert sls.hedges_g(b, a) == pytest.approx(-3.2)


def test_cli_writes_json(tmp_path, monkeypatch):
    csv = tmp_path / "episodes.csv"
    pd.DataFrame(_episodes("x", [0.2, 0.3, 0.4]) + _episodes("y", [0.5, 0.6, 0.7])).to_csv(csv, index=False)
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["seed_level_stats.py", "--csv", str(csv), "--a", "x", "--b", "y",
                                     "--filter", "set=fresh", "--out", str(out)])
    assert sls.main() == 0
    res = json.loads(out.read_text())
    assert res["metrics"]["success"]["difference_of_seed_means"] == pytest.approx(-0.3)
    assert res["metrics"]["type2"]["difference_of_seed_means"] == pytest.approx(0.3)
    assert res["metrics"]["success"]["welch_p"] < 0.05
