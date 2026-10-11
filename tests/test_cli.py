"""
test_cli.py — end-to-end smoke tests of the command-line interface:
train -> benchmark -> plot on a tiny budget.
"""

import json

import pandas as pd
import pytest

from safenav.cli import main


def test_evaluate_baseline_writes_outputs(tmp_path):
    code = main(["evaluate", "--baseline", "gotogoal", "--stages", "beginner", "--episodes", "3",
                 "--out", str(tmp_path)])
    assert code == 0
    df = pd.read_csv(tmp_path / "episodes.csv")
    assert len(df) == 3 and {"success", "collision", "spl", "min_clearance"} <= set(df.columns)
    assert json.loads((tmp_path / "summary.json").read_text())["episodes"] == 3


def test_committed_checkpoints_are_found_and_evaluated(tmp_path):
    """The README commands use results/checkpoints/; the benchmark accepts that layout too."""
    from pathlib import Path

    from safenav.evaluation.benchmark import discover_runs

    root = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    runs = discover_runs(root)
    assert sorted((r["variant"], r["seed"]) for r in runs) == [(v, s) for v in ("filtered", "unfiltered")
                                                               for s in range(5)]
    code = main(["evaluate", "--checkpoint", str(root / "filtered" / "seed_0.pt"), "--stages", "expert",
                 "--episodes", "2", "--out", str(tmp_path)])
    assert code == 0
    df = pd.read_csv(tmp_path / "episodes.csv")
    assert len(df) == 2 and not df["collision"].any()          # filter on by default


def test_invalid_override_is_a_clean_error(capsys):
    assert main(["evaluate", "--baseline", "gotogoal", "--set", "env.dt=-1"]) == 2
    assert "env.dt" in capsys.readouterr().err


@pytest.mark.slow
def test_train_benchmark_plot_pipeline(tmp_path):
    runs = tmp_path / "runs"
    small = ["--set", "ppo.rollout_steps=256", "--set", "ppo.batch_size=64", "--set", "ppo.n_epochs=1",
             "--set", "network.hidden_dims=[16, 16]", "--set", "training.eval_episodes=3",
             "--set", "training.eval_every=256"]
    for variant_flag, variant in (([], "filtered"), (["--no-safety-filter"], "unfiltered")):
        assert main(["train", "--timesteps", "256", "--seed", "0", "--quiet",
                     "--run-dir", str(runs / variant / "seed_0"), *variant_flag, *small]) == 0
    bench = tmp_path / "bench"
    assert main(["benchmark", "--runs", str(runs), "--out", str(bench), "--episodes", "2", "--jobs", "1"]) == 0
    summary = pd.read_csv(bench / "summary.csv")
    # 2 PPO variants x 2 test filter settings + 2 baselines x 2 settings, x 3 stages
    assert len(summary) == 8 * 3
    assert (bench / "trajectories.npz").is_file() and (bench / "scenarios.csv").is_file()
    sweep = tmp_path / "sweep"
    assert main(["safety-sweep", "--out", str(sweep), "--margins", "0.0", "0.3", "--episodes", "2",
                 "--jobs", "1"]) == 0
    figs = tmp_path / "figs"
    assert main(["plot", "--benchmark", str(bench), "--runs", str(runs), "--sweep", str(sweep),
                 "--out", str(figs)]) == 0
    assert (figs / "outcomes_by_condition.png").is_file()
    assert (figs / "training_curves.png").is_file()
    assert (figs / "safety_margin_sweep.png").is_file()
