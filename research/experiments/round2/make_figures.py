"""
make_figures.py — figures for E1a and round 2 (descriptive; reads saved CSV/JSON outputs only).

    python research/experiments/round2/make_figures.py

Writes PDF figures to paper_draft/figures/ (each figure also has a table in the manuscript or in
results/summary.json). Figures whose inputs are missing are skipped and listed.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
FIG = ROOT / "paper_draft" / "figures"
# Categorical slots in fixed order (validated: scripts/validate_palette.js, light mode, all checks pass;
# the contrast warning is relieved by direct labels, hatching and the tables).
C = {"success": "#2a78d6", "type1": "#eb6834", "type2": "#1baf7a", "other": "#eda100"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "pdf.fonttype": 42})


def comp(df):
    """Outcome composition (shares of episodes)."""
    n = len(df)
    t = df[df.outcome == "timeout"]["class"]
    return {"success": (df.outcome == "success").sum() / n, "type1": (t == "filter-blocked").sum() / n,
            "type2": (t == "self-stop").sum() / n, "other": t.isin(["moving", "other stall"]).sum() / n}


def stacked(ax, labels, comps, title):
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    names = {"success": "success", "type1": "Type I (filter-blocked)", "type2": "Type II (self-stop)",
             "other": "other timeout"}
    for k in ("success", "type1", "type2", "other"):
        w = np.array([c[k] for c in comps])
        ax.barh(y, w, left=left, color=C[k], edgecolor=SURF, linewidth=1.0, height=0.62, label=names[k],
                hatch="////" if k == "type1" else None)
        for yi, li, wi in zip(y, left, w):
            if wi >= 0.08:
                ax.text(li + wi / 2, yi, f"{wi * 100:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
        left += w
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100 %"])
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.tick_params(axis="y", length=0)


def fig_e3(missing):
    paths = {s: RES / f"e3_{s}_mean" / "episodes.csv" for s in ("intermediate", "expert")}
    smp = {s: RES / f"e3_{s}_sampled" / "episodes.csv" for s in ("intermediate", "expert")}
    if not all(p.is_file() for p in list(paths.values()) + list(smp.values())):
        missing.append("fig_e3")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    order = [("unfiltered", "mean", "PPO trained w/o filter"), ("filtered", "mean", "PPO trained with filter"),
             ("unfiltered", "sampled", "w/o filter, sampled"), ("filtered", "sampled", "with filter, sampled"),
             ("potential_field", "mean", "potential field"), ("dwa", "mean", "DWA-style (frozen)"),
             ("gotogoal", "mean", "go-to-goal")]
    for ax, s in zip(axes, ("intermediate", "expert")):
        m, sm = pd.read_csv(paths[s]), pd.read_csv(smp[s])
        comps = [comp((m if a == "mean" else sm).query("agent == @ag")) for ag, a, _ in order]
        stacked(ax, [lab for *_, lab in order], comps, f"{s} (1,000 scenarios)")
    axes[0].legend(ncol=4, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "e3_outcomes.pdf")
    plt.close(fig)


def fig_e1a(missing):
    p = ROOT / "research" / "experiments" / "e1a" / "eval" / "episodes.csv"
    if not p.is_file():
        missing.append("fig_e1a")
        return
    df = pd.read_csv(p).query("set == 'fresh' and actions == 'mean'")
    metrics = [("success", lambda d: d.outcome == "success"),
               ("Type I", lambda d: d["class"] == "filter-blocked"),
               ("Type II", lambda d: d["class"] == "self-stop"),
               ("deadlock",
                lambda d: d["class"].isin(["filter-blocked", "self-stop", "other stall"]))]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
    arms = [("committed", "committed\n1M steps"), ("control", "control\n300k"), ("penalty", "penalty α=1\n300k")]
    for ax, (name, f) in zip(axes, metrics):
        for i, (arm, _) in enumerate(arms):
            g = df[df.arm == arm]
            per = g.assign(x=f(g).astype(float)).groupby("train_seed").x.mean()
            jitter = np.linspace(-0.12, 0.12, len(per))
            ax.scatter(i + jitter, per.values, s=14, color=C["success"] if arm != "penalty" else C["type1"],
                       edgecolor=SURF, linewidth=0.8, zorder=3)
            ax.hlines(per.mean(), i - 0.25, i + 0.25, color=INK, linewidth=1.5, zorder=4)
        ax.set_xticks(range(3), [lab for _, lab in arms], fontsize=6)
        ax.set_title(name, fontsize=8, loc="left", color=INK)
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of episodes (per seed)")
    fig.tight_layout()
    fig.savefig(FIG / "e1a_seeds.pdf")
    plt.close(fig)


def fig_dwa(missing):
    tags = [("dwa_test_off", "frozen, filter off"), ("dwa_test_on", "frozen, filter on"),
            ("dwa_margin_0p35_test_on", "margin 0.35 m, filter on"),
            ("dwa_margin_0p45_test_on", "margin 0.45 m, filter on")]
    have = [(t, lab) for t, lab in tags if (RES / t / "episodes.csv").is_file()]
    if len(have) < 2:
        missing.append("fig_dwa")
        return
    fig, axes = plt.subplots(3, 1, figsize=(5.2, 0.55 + 0.42 * 3 * len(have)), sharex=True)
    for ax, stage in zip(axes, ("beginner", "intermediate", "expert")):
        comps = [comp(pd.read_csv(RES / t / "episodes.csv").query("stage == @stage")) for t, _ in have]
        stacked(ax, [lab for _, lab in have], comps, f"{stage} (100 test scenarios)")
    axes[-1].legend(ncol=4, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.3, -0.35))
    fig.tight_layout()
    fig.savefig(FIG / "dwa_filter.pdf")
    plt.close(fig)


def fig_e4(missing):
    p = RES / "e4_mean" / "episodes.csv"
    if not p.is_file():
        missing.append("fig_e4")
        return
    df = pd.read_csv(p)
    arms = [("control", "control"), ("riskfree", "risk penalty off"), ("beta_policy", "Beta policy")]
    metrics = [("success", lambda d: d.outcome == "success"),
               ("Type I", lambda d: d["class"] == "filter-blocked"),
               ("policy-side stall (primary)", lambda d: d["class"].isin(["self-stop", "other stall"]))]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for ax, (name, f) in zip(axes, metrics):
        for i, (arm, _) in enumerate(arms):
            g = df[df.agent == arm]
            per = g.assign(x=f(g).astype(float)).groupby("train_seed").x.mean()
            jitter = np.linspace(-0.12, 0.12, len(per))
            ax.scatter(i + jitter, per.values, s=14, color=[C["success"], C["type2"], C["type1"]][i],
                       edgecolor=SURF, linewidth=0.8, zorder=3)
            ax.hlines(per.mean(), i - 0.25, i + 0.25, color=INK, linewidth=1.5, zorder=4)
        ax.set_xticks(range(3), [lab for _, lab in arms], fontsize=6.5)
        ax.set_title(name, fontsize=8, loc="left", color=INK)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of episodes (per seed)")
    fig.tight_layout()
    fig.savefig(FIG / "e4_seeds.pdf")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    missing = []
    for f in (fig_e1a, fig_e3, fig_dwa, fig_e4):
        f(missing)
    print(json.dumps({"missing": missing}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
