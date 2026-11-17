"""
make_figures_r4.py — figures for round 4 (reads results/ and timeouts/ only).

    python research/experiments/round4/make_figures_r4.py

Writes paper_draft/figures/r4_latency.pdf, r4_robustness.pdf, r4_fairness.pdf, r4_timeouts.pdf.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = ROOT / "paper_draft" / "figures"
RES = HERE / "results"
C = {"success": "#2a78d6", "type1": "#eb6834", "type2": "#1baf7a", "other": "#eda100"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42})
STALLS = ("filter-blocked", "self-stop", "other stall")


def comp(g):
    n, c = len(g), g["class"].fillna("")
    return {"success": (g.outcome == "success").mean(), "type1": (c == "filter-blocked").sum() / n,
            "type2": c.isin(["self-stop", "other stall"]).sum() / n, "other": ((c == "moving") | (g.outcome == "collision")).sum() / n}


def stacked(ax, labels, comps, title):
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    names = {"success": "success", "type1": "filter-blocked", "type2": "self-stop / other stall", "other": "moving timeout or collision"}
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
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.tick_params(axis="y", length=0)


def fig_latency():
    p = RES / "latency" / "calls.csv"
    if not p.is_file():
        return "latency missing"
    c = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for pl, col in (("reference", C["type1"]), ("fast", C["success"])):
        x = np.sort(c[c.planner == pl].wall_ms.to_numpy())
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=col, linewidth=1.4, label=f"{pl} (p99 {np.percentile(x, 99):.1f} ms)")
    ax.axvline(100, color=MUTED, linestyle="--", linewidth=0.8)
    ax.text(105, 0.3, "100 ms target", fontsize=6.5, color=MUTED)
    ax.axhline(0.01, color=MUTED, linestyle=":", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("planner call latency, wall clock (ms)")
    ax.set_ylabel("share of calls slower")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "r4_latency.pdf")
    plt.close(fig)


def fig_robustness():
    conds = [("expert, exact (C)", "r_expert", "C"), ("localisation error", "r_expert", "C_loc"),
             ("map error", "r_expert", "C_map"), ("sensed map", "r_expert", "C_sensed"),
             ("intermediate", "r_intermediate", "C"), ("dense", "r_dense", "C"), ("random heading", "r_heading", "C")]
    have = {t: pd.read_csv(RES / t / "episodes.csv") for t in {c[1] for c in conds} if (RES / t / "episodes.csv").is_file()}
    if not have:
        return "robustness missing"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (agent, title) in zip(axes, (("filtered", "trained with the filter"), ("unfiltered", "trained without the filter"))):
        labels, comps = [], []
        for lab, tag, meth in conds:
            if tag not in have:
                continue
            d = have[tag]
            if lab.startswith("expert") or tag != "r_expert":
                labels.append(f"{lab.split(' (')[0]}: baseline")
                comps.append(comp(d[(d.method == "baseline") & (d.agent == agent)]))
            labels.append(f"{lab.split(' (')[0]}: {meth}")
            comps.append(comp(d[(d.method == meth) & (d.agent == agent)]))
        stacked(ax, labels, comps, title)
    axes[0].legend(ncol=4, fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    fig.savefig(FIG / "r4_robustness.pdf")
    plt.close(fig)


def fig_fairness():
    p = RES / "r_expert" / "episodes.csv"
    if not p.is_file():
        return "fairness missing"
    d = pd.read_csv(p)
    rows = [("nofilter", "policy, no filter"), ("baseline", "policy + filter"), ("A", "A: escape planning"),
            ("C", "C: triggered planning"), ("planner_only", "planner only (no policy)"),
            ("path_follower", "shortest-path follower"), ("path_follower_inflated", "path follower, +0.2 m (expl.)")]
    labels, comps = [], []
    for m, lab in rows:
        g = d[d.method == m]
        if len(g):
            labels.append(lab)
            comps.append(comp(g))
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    stacked(ax, labels, comps, "expert, 200 scenarios (learned agents: 10 agents pooled)")
    ax.legend(ncol=2, fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "r4_fairness.pdf")
    plt.close(fig)


def fig_timeouts():
    p = HERE / "timeouts" / "counterfactuals.csv"
    if not p.is_file():
        return "timeouts missing"
    cf = pd.read_csv(p)
    order = [("control", "unchanged (control)"), ("max_steps_800", "time limit 800 steps"), ("window_15", "trigger window 15 steps"),
             ("no_handback", "recovery keeps control"), ("planner_only", "planner only"), ("gamma_1.0", "filter gamma 1.0 (diagnostic)")]
    labels = [lab for v, lab in order if v in set(cf.variant)]
    comps = [comp(cf[cf.variant == v]) for v, _ in order if v in set(cf.variant)]
    fig, ax = plt.subplots(figsize=(5.0, 2.2))
    stacked(ax, labels, comps, "200 sampled moving timeouts of C (round 3), one factor changed at a time")
    ax.legend(ncol=2, fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    fig.savefig(FIG / "r4_timeouts.pdf")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    print([f() for f in (fig_latency, fig_robustness, fig_fairness, fig_timeouts)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
