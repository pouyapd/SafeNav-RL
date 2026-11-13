"""
make_figures_r3.py — figures for round 3 (reads results/<tag>/episodes.csv; replays a few episodes for trajectories).

    python research/experiments/round3/make_figures_r3.py [--tag main]

Writes paper_draft/figures/r3_outcomes.pdf, r3_seeds.pdf and r3_trajectories.pdf.
Trajectory cases are chosen by rule, not by appearance: for each agent type, the scenarios with the
lowest seed in which the baseline deadlocks (seed 0 agent), plus the lowest-seed scenario in which C
ends in a "moving" timeout, if any.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "paper_draft" / "figures"
C = {"success": "#2a78d6", "type1": "#eb6834", "type2": "#1baf7a", "other": "#eda100"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42})
ORDER = [("baseline", "baseline"), ("A", "A: escape planning"), ("B", "B: triggered behaviour"),
         ("C", "C: triggered planning"), ("C_euclid", "C, straight-line cost")]
TYPES = [("filtered", "trained with the filter"), ("unfiltered", "trained without the filter")]


def comp(g):
    n, c = len(g), g["class"].fillna("")
    return {"success": (g.outcome == "success").mean(), "type1": (c == "filter-blocked").sum() / n,
            "type2": c.isin(["self-stop", "other stall"]).sum() / n, "other": (c == "moving").sum() / n}


def fig_outcomes(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    names = {"success": "success", "type1": "filter-blocked", "type2": "self-stop / other stall", "other": "moving timeout"}
    for ax, (agent, title) in zip(axes, TYPES):
        methods = [(m, lab) for m, lab in ORDER if m in set(df.method)]
        y = np.arange(len(methods))[::-1]
        left = np.zeros(len(methods))
        comps = [comp(df[(df.method == m) & (df.agent == agent)]) for m, _ in methods]
        for k in ("success", "type1", "type2", "other"):
            w = np.array([c[k] for c in comps])
            ax.barh(y, w, left=left, color=C[k], edgecolor=SURF, linewidth=1.0, height=0.62, label=names[k],
                    hatch="////" if k == "type1" else None)
            for yi, li, wi in zip(y, left, w):
                if wi >= 0.07:
                    ax.text(li + wi / 2, yi, f"{wi * 100:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
            left += w
        ax.set_yticks(y, [lab for _, lab in methods])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100 %"])
        ax.set_title(title, fontsize=8, loc="left", color=INK)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(ncol=4, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13))
    fig.tight_layout()
    fig.savefig(out / "r3_outcomes.pdf")
    plt.close(fig)


def fig_seeds(df, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    methods = [m for m, _ in ORDER if m in set(df.method)]
    for col, (agent, tlabel) in enumerate(TYPES):
        for row, (metric, f) in enumerate((("deadlock", lambda g: g["class"].isin(["filter-blocked", "self-stop", "other stall"])),
                                           ("success", lambda g: g.outcome == "success"))):
            ax = axes[2 * col + row]
            for i, m in enumerate(methods):
                g = df[(df.method == m) & (df.agent == agent)]
                per = g.assign(x=f(g).astype(float)).groupby("train_seed").x.mean()
                ax.scatter(i + np.linspace(-0.12, 0.12, len(per)), per.values, s=12,
                           color=[C["type1"], C["success"], C["type2"], C["other"], MUTED][i], edgecolor=SURF, linewidth=0.6, zorder=3)
                ax.hlines(per.mean(), i - 0.25, i + 0.25, color=INK, linewidth=1.4, zorder=4)
            ax.set_xticks(range(len(methods)), [m.replace("_euclid", "\neucl.") for m in methods], fontsize=6)
            ax.set_ylim(0, 1)
            ax.set_title(f"{metric}, {'with' if agent == 'filtered' else 'w/o'} filter", fontsize=7.5, loc="left", color=INK)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
    axes[0].set_ylabel("share (per seed)")
    fig.tight_layout()
    fig.savefig(out / "r3_seeds.pdf")
    plt.close(fig)


def replay(agent_dir, method, seed, ep_seed):
    from safenav.agents.policy import PPOPolicy
    from safenav.config import get_stage, load_config
    from safenav.factory import make_env
    from safenav.recovery import RecoveryConfig, RecoveryController
    config = load_config(ROOT / "configs" / "default_config.yaml", overrides={"env.goal_sampling": "rejection"})
    env = make_env(config, task=get_stage(config, "expert"), safety_filter=True)
    policy = PPOPolicy.from_checkpoint(ROOT / "results" / "checkpoints" / agent_dir / f"seed_{seed}.pt", deterministic=True)
    spec = {"baseline": None, "C": ("C", "geodesic")}[method]
    ctl = RecoveryController(policy, RecoveryConfig(mode=spec[0], cost=spec[1])) if spec else None
    obs, info = env.reset(seed=ep_seed)
    if ctl:
        ctl.reset(seed=ep_seed)
        ctl.bind(env)
    rec, term, trunc = [], False, False
    while not (term or trunc):
        a = (ctl or policy).act(obs)
        rec.append(bool(ctl.in_recovery) if ctl else False)
        obs, _, term, trunc, info = env.step(a)
    base = env.unwrapped
    return np.asarray(base.trajectory), np.asarray(rec), base.obstacle_map.obstacles, base.goal_pos.copy(), info


def fig_trajectories(df, out):
    cases = []
    for agent, _ in TYPES:
        b = df[(df.method == "baseline") & (df.agent == agent) & (df.train_seed == 0)]
        dl = b[b["class"].isin(["filter-blocked", "self-stop"])].sort_values("episode_seed")
        cases += [(agent, int(s)) for s in dl.episode_seed.head(2)]
    mv = df[(df.method == "C") & (df["class"] == "moving") & (df.train_seed == 0)].sort_values("episode_seed")
    if len(mv):
        cases.append((mv.iloc[0].agent, int(mv.iloc[0].episode_seed)))
    base_seed = int(df.episode_seed.min())
    fig, axes = plt.subplots(1, len(cases), figsize=(1.75 * len(cases), 2.0))
    for ax, (agent, ep) in zip(np.atleast_1d(axes), cases):
        for method, col, lw in (("baseline", MUTED, 1.0), ("C", C["success"], 1.2)):
            traj, rec, obs, goal, info = replay(agent, method, 0, ep)
            ax.plot(traj[:, 0], traj[:, 1], color=col, linewidth=lw, label=method)
            if method == "baseline":
                ax.plot(*traj[-1], "x", color=INK, markersize=5, markeredgewidth=1.2, zorder=5,
                        label="baseline stops" if ax is np.atleast_1d(axes)[0] else None)
            if method == "C" and rec.any():
                r = np.concatenate([[False], rec])
                ax.scatter(traj[r, 0], traj[r, 1], s=1.5, color=C["type1"], zorder=4, label="recovery active")
        for o in obs:
            ax.add_patch(plt.Circle((o.x, o.y), o.radius, color="#c9c8c3", zorder=1))
            ax.add_patch(plt.Circle((o.x, o.y), o.radius + 0.6, fill=False, linestyle=":", linewidth=0.5, color=MUTED))  # keep-out: radius + robot + margin
        ax.plot(*traj[0], "o", color=INK, markersize=3)
        ax.add_patch(plt.Circle(goal, 0.5, fill=False, color=C["type2"], linewidth=1))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{'with' if agent == 'filtered' else 'w/o'} filter, scenario {ep - base_seed}", fontsize=6.5, loc="left")
    np.atleast_1d(axes)[0].legend(fontsize=5.5, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.45), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "r3_trajectories.pdf")
    plt.close(fig)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(FIG))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(HERE / "results" / args.tag / "episodes.csv")
    fig_outcomes(df, out)
    fig_seeds(df, out)
    print("trajectory cases:", fig_trajectories(df, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
