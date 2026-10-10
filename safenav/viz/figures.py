"""
figures.py — README figures generated from real outputs.

Inputs:
  benchmark_dir  output of `python -m safenav benchmark`
                 (summary.csv, episodes.csv, scenarios.csv, trajectories.npz, metadata.json)
  runs_dir       output of `python -m safenav train-many` (progress.csv, validation.csv per run)
  sweep_dir      output of `python -m safenav safety-sweep` (safety_sweep.csv)

Every figure states its data source in a footnote. Nothing here invents or
smooths numbers beyond what the caption says.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from safenav.agents.policy import PPOPolicy
from safenav.config import get_stage
from safenav.env.obstacle_map import Obstacle
from safenav.evaluation.benchmark import STAGES, load_trajectories
from safenav.evaluation.runner import run_episode
from safenav.factory import make_env
from safenav.safety.cbf_filter import CBFSafetyFilter
from safenav.viz import style
from safenav.viz.style import INK, INK_2, MUTED, OUTCOME_COLORS, apply_style, finish, method_color

PPO_F = "PPO (filter in training)"
PPO_U = "PPO (no filter in training)"
STAGE_TITLES = {"beginner": "Beginner (2-4 obstacles)", "intermediate": "Intermediate (4-8 obstacles)",
                "expert": "Expert (8-12 obstacles)"}


# ── Scenario drawing ──────────────────────────────────────────────────────────

def draw_scenario(ax, obstacles, goal, start, workspace: float = 10.0, robot_radius: float = 0.3,
                  margin: Optional[float] = None, goal_radius: float = 0.5, endpoints: bool = True) -> None:
    ax.set_xlim(0, workspace)
    ax.set_ylim(0, workspace)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])
    for (x, y, r) in obstacles:
        ax.add_patch(mpatches.Circle((x, y), r, facecolor=style.OBSTACLE_FILL, edgecolor=style.OBSTACLE_EDGE,
                                     linewidth=0.8, zorder=2))
        if margin is not None and margin > 0:
            ax.add_patch(mpatches.Circle((x, y), r + robot_radius + margin, fill=False,
                                         edgecolor=style.MARGIN_EDGE, linewidth=0.7, linestyle=(0, (2, 2)),
                                         zorder=2))
    if not endpoints:
        return
    ax.add_patch(mpatches.Circle(goal, goal_radius, facecolor="#d9f2d9", edgecolor=OUTCOME_COLORS["success"],
                                 linewidth=1.0, zorder=2))
    ax.plot(*goal, marker="*", markersize=12, color=OUTCOME_COLORS["success"], zorder=4,
            markeredgecolor=style.SURFACE, markeredgewidth=0.8)
    ax.plot(*start, marker="o", markersize=7, color=INK, zorder=5,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5)


def draw_path(ax, xy, color, outcome: str, interventions=None, label: Optional[str] = None) -> None:
    xy = np.asarray(xy)
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, zorder=3, label=label)
    if interventions is not None and len(interventions) and np.any(interventions):
        pts = xy[1:][np.asarray(interventions, dtype=bool)]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, color=INK_2, alpha=0.8, zorder=4, linewidths=0)
    end_marker = {"success": "o", "timeout": "s", "collision": "X", "out_of_bounds": "X"}.get(outcome, "o")
    end_color = OUTCOME_COLORS.get("collision" if outcome == "out_of_bounds" else outcome, MUTED)
    ax.plot(*xy[-1], marker=end_marker, markersize=8, color=end_color, zorder=6,
            markeredgecolor=style.SURFACE, markeredgewidth=1.5)


def _outcome_text(outcome: str) -> str:
    return {"success": "reached goal", "timeout": "timeout", "collision": "collision",
            "out_of_bounds": "left workspace"}.get(outcome, outcome)


# ── Figure 1: same scenario, four controllers ────────────────────────────────

HERO_CONDITIONS = [
    (PPO_F, True, "PPO trained with filter · filter on"),
    (PPO_U, False, "PPO trained without filter · no filter"),
    ("PotentialField", True, "Potential field · filter on"),
    ("GoToGoal", False, "Go-to-goal · no filter"),
]


def fig_hero(benchmark_dir: Path, config: Dict[str, Any], out: Path, n_scenarios: int = 3) -> Optional[Path]:
    """Grid: the first n expert test scenarios (no selection) x the four conditions in HERO_CONDITIONS."""
    trajs = load_trajectories(benchmark_dir / "trajectories.npz")
    by_key = {(t["method"], bool(t["safety_filter"]), t["stage"], int(t["episode_seed"])): t for t in trajs}
    seeds = sorted({int(t["episode_seed"]) for t in trajs if t["stage"] == "expert"})[:n_scenarios]
    if not seeds or not all(by_key.get((m, f, "expert", s)) for s in seeds for m, f, _ in HERO_CONDITIONS):
        return None
    margin = float(config["safety"]["safe_distance"])
    train_seed = next((t["train_seed"] for t in trajs if t["train_seed"] not in (None, "None")), "-")
    fig, axes = plt.subplots(len(seeds), 4, figsize=(12.5, 3.25 * len(seeds)))
    for r, seed in enumerate(seeds):
        for c, (method, filt, title) in enumerate(HERO_CONDITIONS):
            ax = axes[r, c]
            t = by_key[(method, filt, "expert", seed)]
            draw_scenario(ax, t["obstacles"], t["goal"], t["trajectory"][0], margin=margin if filt else None)
            draw_path(ax, t["trajectory"], method_color(method), t["outcome"],
                      interventions=t["interventions"] if filt else None)
            ax.text(0.03, 0.97, f"{_outcome_text(t['outcome'])}, {len(t['trajectory']) - 1} steps",
                    transform=ax.transAxes, fontsize=8.5, va="top", color=INK_2,
                    bbox=dict(facecolor=style.SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"scenario {seed}", fontsize=9)
    handles = [
        Line2D([], [], marker="o", color=INK, linestyle="", markersize=6, label="start"),
        Line2D([], [], marker="*", color=OUTCOME_COLORS["success"], linestyle="", markersize=10, label="goal"),
        Line2D([], [], color=style.MARGIN_EDGE, linestyle=(0, (2, 2)), label="keep-out boundary (filter on)"),
        Line2D([], [], marker="o", color=INK_2, linestyle="", markersize=3, label="filter reduced speed"),
        Line2D([], [], marker="X", color=OUTCOME_COLORS["collision"], linestyle="", markersize=7, label="collision"),
        Line2D([], [], marker="s", color=OUTCOME_COLORS["timeout"], linestyle="", markersize=6, label="timeout"),
    ]
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    path = out / "hero_same_scenario.png"
    finish(fig, path, note=f"The first {len(seeds)} expert-stage test scenarios (8-12 obstacles), no selection; each row is "
                           f"one scenario, identical for every controller. PPO agents: training seed {train_seed}.",
           note_y=-0.035)
    return path


# ── Figure 2: outcome composition ────────────────────────────────────────────

CONDITION_ROWS = [
    (PPO_F, True), (PPO_F, False), (PPO_U, True), (PPO_U, False),
    ("PotentialField", True), ("PotentialField", False), ("GoToGoal", True), ("GoToGoal", False),
]


def _row_label(method: str, filt: bool) -> str:
    short = {PPO_F: "PPO, trained with filter", PPO_U: "PPO, trained without filter"}.get(method, method)
    return f"{short} · test filter {'on' if filt else 'off'}"


def fig_outcomes(summary: pd.DataFrame, out: Path, episodes_per_stage: int, n_seeds: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    rows = [r for r in CONDITION_ROWS if ((summary.method == r[0]) & (summary.safety_filter == r[1])).any()]
    y = np.arange(len(rows))[::-1]
    for ax, stage in zip(axes, STAGES):
        for yi, (method, filt) in zip(y, rows):
            rec = summary[(summary.method == method) & (summary.safety_filter == filt) & (summary.stage == stage)].iloc[0]
            left = 0.0
            for key in ("success", "timeout", "collision"):
                val = float(rec[f"{key}_rate"]) * 100
                if val <= 0:
                    continue
                ax.barh(yi, val - 0.4 if val > 0.6 else val, left=left, height=0.62,
                        color=OUTCOME_COLORS[key], linewidth=0)
                if val >= 9:
                    ax.text(left + val / 2, yi, f"{val:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if key != "timeout" else INK)
                left += val
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of test episodes (%)")
        ax.set_title(STAGE_TITLES[stage], fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_row_label(m, f) for m, f in rows])
    for tick, (m, _) in zip(axes[0].get_yticklabels(), rows):
        tick.set_color(INK)
    handles = [mpatches.Patch(color=OUTCOME_COLORS[k], label=k) for k in ("success", "timeout", "collision")]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.04))
    fig.tight_layout()
    path = out / "outcomes_by_condition.png"
    finish(fig, path, note=f"Benchmark summary.csv. {episodes_per_stage} test episodes per stage (identical scenarios "
                           f"for every row). PPO rows: mean over {n_seeds} training seeds. Collision includes leaving "
                           f"the workspace. Labels shown for segments >= 9%.")
    return path


# ── Figure 3: safety vs task success (expert) ────────────────────────────────

def fig_tradeoff(summary: pd.DataFrame, out: Path, stage: str = "expert") -> Path:
    sub = summary[summary.stage == stage]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for method in style.METHOD_ORDER:
        for filt in (False, True):
            rec = sub[(sub.method == method) & (sub.safety_filter == filt)]
            if rec.empty:
                continue
            rec = rec.iloc[0]
            x, y = rec["collision_rate"] * 100, rec["success_rate"] * 100
            xerr = [[x - rec["collision_ci_low"] * 100], [rec["collision_ci_high"] * 100 - x]]
            yerr = [[y - rec["success_ci_low"] * 100], [rec["success_ci_high"] * 100 - y]]
            color = method_color(method)
            ax.errorbar(x, y, xerr=np.clip(xerr, 0, None), yerr=np.clip(yerr, 0, None), fmt="none",
                        ecolor=color, elinewidth=1.0, capsize=0, alpha=0.7, zorder=2)
            ax.plot(x, y, marker="o", markersize=9, linestyle="",
                    markerfacecolor=color if filt else style.SURFACE, markeredgecolor=color,
                    markeredgewidth=2, zorder=3)
        both = sub[sub.method == method]
        if len(both) == 2:
            on, off = both[both.safety_filter].iloc[0], both[~both.safety_filter].iloc[0]
            ax.annotate("", xy=(on.collision_rate * 100, on.success_rate * 100),
                        xytext=(off.collision_rate * 100, off.success_rate * 100),
                        arrowprops=dict(arrowstyle="-|>", color=method_color(method), lw=1.0, alpha=0.6,
                                        shrinkA=7, shrinkB=7), zorder=1)
    ax.set_xlabel("collision rate (%)  ← safer")
    ax.set_ylabel("success rate (%)  higher is better →")
    ax.set_xlim(left=-2)
    ax.set_ylim(top=102)
    ax.set_title(f"{STAGE_TITLES[stage]}: effect of the safety filter")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=method_color(m), label=m)
               for m in style.METHOD_ORDER if (sub.method == m).any()]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=MUTED,
                       markeredgecolor=MUTED, label="filter on (filled)"),
                Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=style.SURFACE,
                       markeredgecolor=MUTED, markeredgewidth=2, label="filter off (hollow)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    path = out / "safety_vs_success_expert.png"
    n_seeds = int(sub["train_seeds"].max()) if "train_seeds" in sub else 0
    finish(fig, path, note=f"Arrows: filter off -> on. Whiskers: 95% CI (t-interval over {n_seeds} training seeds "
                           f"for PPO; Wilson interval for the baselines).")
    return path


# ── Figure 4: training curves ────────────────────────────────────────────────

def _load_runs(runs_dir: Path, name: str) -> Dict[str, List[pd.DataFrame]]:
    out: Dict[str, List[pd.DataFrame]] = {}
    for variant in ("filtered", "unfiltered"):
        frames = [pd.read_csv(p) for p in sorted((runs_dir / variant).glob(f"seed_*/{name}")) if p.stat().st_size > 0]
        # The final validation can coincide with the last periodic one: keep one row per timestep.
        frames = [f.drop_duplicates("timestep", keep="last") for f in frames if len(f)]
        if frames:
            out[variant] = frames
    return out


def fig_training(runs_dir: Path, out: Path) -> Optional[Path]:
    val = _load_runs(runs_dir, "validation.csv")
    prog = _load_runs(runs_dir, "progress.csv")
    if not val:
        return None
    label = {"filtered": PPO_F, "unfiltered": PPO_U}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for variant, frames in val.items():
        color = method_color(label[variant])
        grid = np.array(sorted(set.intersection(*[set(f.timestep) for f in frames])))
        mat = np.vstack([f.set_index("timestep").loc[grid, "success_rate"].to_numpy() for f in frames]) * 100
        axes[0].fill_between(grid / 1e6, mat.min(0), mat.max(0), color=color, alpha=0.12, linewidth=0)
        axes[0].plot(grid / 1e6, mat.mean(0), color=color, label=f"{label[variant]} (n={len(frames)} seeds)")
    axes[0].set_title("Validation success (all stages, filter as in training)")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 100)
    for variant, frames in prog.items():
        color = method_color(label[variant])
        grid = np.linspace(0, min(f.timestep.max() for f in frames), 200)
        mat = np.vstack([np.interp(grid, f.timestep, f.collision_rate_100) for f in frames]) * 100
        axes[1].fill_between(grid / 1e6, mat.min(0), mat.max(0), color=color, alpha=0.12, linewidth=0)
        axes[1].plot(grid / 1e6, mat.mean(0), color=color, label=label[variant])
    axes[1].set_title("Collisions during training (last 100 episodes)")
    axes[1].set_ylabel("collision rate (%)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    path = out / "training_curves.png"
    finish(fig, path, note="Line: mean over training seeds; band: min-max over seeds. Training episodes follow the "
                           "curriculum, so the collision rate mixes difficulty stages.")
    return path


# ── Figure 5: safety-margin sweep ────────────────────────────────────────────

def fig_margin_sweep(sweep: pd.DataFrame, out: Path, chosen_margin: float) -> Path:
    ramp = {"beginner": style.SEQ_BLUE[2], "intermediate": style.SEQ_BLUE[4], "expert": style.SEQ_BLUE[6]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    feas = sweep.drop_duplicates(["margin", "stage"])
    for stage in STAGES:
        d = feas[feas.stage == stage].sort_values("margin")
        axes[0].plot(d.margin, d.feasible_rate * 100, marker="o", markersize=5, color=ramp[stage],
                     label=STAGE_TITLES[stage])
        pf = sweep[(sweep.stage == stage) & (sweep.method == "PotentialField")].sort_values("margin")
        axes[1].plot(pf.margin, pf.success_rate * 100, marker="o", markersize=5, color=ramp[stage],
                     label=STAGE_TITLES[stage])
    for ax in axes:
        ax.axvline(chosen_margin, color=MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
        ax.text(chosen_margin + 0.01, 3, f"default {chosen_margin:.1f} m", fontsize=8, color=INK_2)
        ax.set_xlabel("safety margin d_safe (m)")
        ax.set_ylim(0, 103)
    axes[0].set_title("Goal reachable while keeping the margin (oracle)")
    axes[0].set_ylabel("test scenarios (%)")
    axes[1].set_title("Potential-field controller with the filter: success")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    path = out / "safety_margin_sweep.png"
    finish(fig, path, note="Oracle: grid search (5 cm) in the free space inflated by robot radius + margin. "
                           "Same 100 test scenarios per stage at every margin. No agent was retrained.")
    return path


# ── Figure 6: where the policy struggles ─────────────────────────────────────

def fig_difficulty(episodes: pd.DataFrame, scenarios: pd.DataFrame, out: Path) -> Path:
    df = episodes.merge(scenarios[["stage", "episode_seed", "n_obstacles", "reachable_with_margin"]],
                        on=["stage", "episode_seed"])
    df = df[df.safety_filter]
    bins = [0, 4, 7, 10, 13, 16]
    labels = ["2-4", "5-7", "8-10", "11-13", "14-15"]
    df["obstacle_bin"] = pd.cut(df.n_obstacles, bins=bins, labels=labels, right=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, metric, title in ((axes[0], "success", "Success rate"), (axes[1], "timeout", "Timeout rate")):
        for method in style.METHOD_ORDER:
            d = df[df.method == method]
            if d.empty:
                continue
            g = d.groupby("obstacle_bin", observed=False)[metric].agg(["mean", "count"])
            g = g[g["count"] >= 20]
            ax.plot(range(len(g)), g["mean"] * 100, marker="o", markersize=6, color=method_color(method),
                    label=method)
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(g.index.astype(str))
        ax.set_title(f"{title} with the filter on, by obstacle count")
        ax.set_xlabel("obstacles in the scenario")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("share of episodes (%)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    path = out / "difficulty_obstacles.png"
    finish(fig, path, note="All stages pooled; PPO pooled over 5 training seeds. Bins with fewer than 20 episodes "
                           "are not shown.")
    return path


def fig_failures(benchmark_dir: Path, config: Dict[str, Any], out: Path, max_cases: int = 6) -> Optional[Path]:
    """Replay (deterministically) failed expert episodes of the seed-0 PPO agent trained with the filter."""
    episodes = pd.read_csv(benchmark_dir / "episodes.csv")
    scenarios = pd.read_csv(benchmark_dir / "scenarios.csv")
    import json
    meta = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
    runs = [r for r in meta.get("runs", []) if r["variant"] == "filtered"]
    if not runs:
        return None
    run = min(runs, key=lambda r: r["seed"])
    fails = episodes[(episodes.method == PPO_F) & (episodes.safety_filter) & (episodes.train_seed == run["seed"])
                     & (episodes.stage == "expert") & (~episodes.success)].head(max_cases)
    if fails.empty:
        return None
    ckpt = Path(run["checkpoint"])
    if not ckpt.is_file():   # benchmark run elsewhere: use the copies committed under results/checkpoints
        ckpt = benchmark_dir.parent / "checkpoints" / run["variant"] / f"seed_{run['seed']}.pt"
    policy = PPOPolicy.from_checkpoint(ckpt)
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    margin = float(config["safety"]["safe_distance"])
    n = len(fails)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, fails.iterrows()):
        r = run_episode(env, policy, int(row.episode_seed), keep_trajectory=True)
        draw_scenario(ax, r.obstacles, r.goal, r.trajectory[0], margin=margin)
        draw_path(ax, r.trajectory, method_color(PPO_F), r.outcome, interventions=r.interventions_mask)
        sc = scenarios[(scenarios.stage == "expert") & (scenarios.episode_seed == row.episode_seed)].iloc[0]
        reach = "reachable" if sc.reachable_with_margin else "not reachable"
        ax.set_title(f"seed {int(row.episode_seed)}: {_outcome_text(r.outcome)}\n(goal {reach} with margin)",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = out / "failure_cases_expert.png"
    finish(fig, path, note=f"All failed expert test episodes of the PPO agent trained with the filter (training seed "
                           f"{run['seed']}), up to {max_cases}, replayed deterministically from their seeds.")
    return path


# ── Figure 7: what the filter computes ───────────────────────────────────────

def fig_speed_limit(config: Dict[str, Any], out: Path) -> Path:
    flt = CBFSafetyFilter.from_config(config)
    obstacles = [Obstacle(4.0, 5.5, 0.8), Obstacle(6.8, 3.8, 0.5), Obstacle(6.5, 7.4, 0.6)]
    xs = np.linspace(0, 10, 300)
    headings = [(0.0, "heading east (→)"), (np.pi / 2, "heading north (↑)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", style.SEQ_BLUE[::-1])
    for ax, (theta, title) in zip(axes, headings):
        grid = np.zeros((len(xs), len(xs)))
        for i, y in enumerate(xs):
            for j, x in enumerate(xs):
                h, _ = flt.barriers((x, y), theta, obstacles)
                grid[i, j] = flt.speed_limit((x, y), theta, obstacles) if h.min() >= 0 else np.nan
        im = ax.imshow(grid, origin="lower", extent=(0, 10, 0, 10), cmap=cmap, vmin=0, vmax=1)
        draw_scenario(ax, [(o.x, o.y, o.radius) for o in obstacles], (9.0, 9.0), (1.0, 1.0),
                      margin=flt.safe_distance, endpoints=False)
        ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("largest allowed speed v (m/s); darker = slower")
    cb.outline.set_visible(False)
    path = out / "cbf_speed_limit.png"
    finish(fig, path, note=f"Computed with the implemented filter (margin {flt.safe_distance} m, wall margin "
                           f"{flt.wall_margin} m, gamma {flt.gamma}/s, dt {flt.dt} s). White: outside the safe set.")
    return path


def make_all_figures(config: Dict[str, Any], benchmark_dir: Optional[Path], runs_dir: Optional[Path],
                     sweep_dir: Optional[Path], out_dir: Path) -> List[Path]:
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Optional[Path]] = [fig_speed_limit(config, out_dir)]
    if benchmark_dir is not None:
        summary = pd.read_csv(benchmark_dir / "summary.csv")
        episodes = pd.read_csv(benchmark_dir / "episodes.csv")
        scenarios = pd.read_csv(benchmark_dir / "scenarios.csv")
        import json
        meta = json.loads((benchmark_dir / "metadata.json").read_text(encoding="utf-8"))
        n_seeds = max([sum(r["variant"] == v for r in meta.get("runs", [])) for v in ("filtered", "unfiltered")])
        written += [fig_hero(benchmark_dir, config, out_dir),
                    fig_outcomes(summary, out_dir, int(meta["episodes_per_stage"]), n_seeds),
                    fig_tradeoff(summary, out_dir), fig_difficulty(episodes, scenarios, out_dir),
                    fig_failures(benchmark_dir, config, out_dir)]
    if runs_dir is not None:
        written.append(fig_training(runs_dir, out_dir))
    if sweep_dir is not None:
        sweep = pd.read_csv(sweep_dir / "safety_sweep.csv")
        written.append(fig_margin_sweep(sweep, out_dir, float(config["safety"]["safe_distance"])))
    return [p for p in written if p is not None]
