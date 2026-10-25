"""
paper_figures_stats.py

Builds the new figures and the statistics used in paper_draft/manuscript.tex from
stored outputs only (no simulation):
  research/audit_outputs/stall_taxonomy_all.csv   (research/scripts/stall_taxonomy.py)
  research/audit_outputs/paired_comparisons_expert.csv
  results/benchmark/episodes.csv, results/stochastic_eval/episodes.csv

    python research/scripts/paper_figures_stats.py      # seconds

Outputs: paper_draft/figures/{stall_taxonomy,stall_geometry,paired_differences}.pdf
         paper_draft/generated/stats.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_diagnostics import paired_hier_bootstrap  # noqa: E402

FIG = ROOT / "paper_draft" / "figures"
GEN = ROOT / "paper_draft" / "generated"
# Validated categorical slots 1-4 (dataviz reference palette, light mode); values are printed on the bars.
COLORS = {"filter-blocked": "#2a78d6", "self-stop": "#eb6834", "moving": "#1baf7a", "other stall": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASSES = ["filter-blocked", "self-stop", "moving", "other stall"]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def conditions(tax):
    """Ordered (label, selector) for the plotted conditions."""
    return [
        ("PPO trained w/o filter, mean", (tax.policy == "ppo") & (tax.variant == "unfiltered") & (tax.actions == "mean")),
        ("PPO trained w/o filter, sampled", (tax.policy == "ppo") & (tax.variant == "unfiltered") & (tax.actions == "sampled")),
        ("PPO trained with filter, mean", (tax.policy == "ppo") & (tax.variant == "filtered") & (tax.actions == "mean")),
        ("PPO trained with filter, sampled", (tax.policy == "ppo") & (tax.variant == "filtered") & (tax.actions == "sampled")),
        ("Go-to-goal", tax.policy == "gotogoal"),
        ("Potential field", tax.policy == "potential_field"),
    ]


def fig_taxonomy(tax, stats):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, stage in zip(axes, ("intermediate", "expert")):
        conds = conditions(tax)
        labels = [c[0] for c in conds]
        y = np.arange(len(conds))[::-1]
        stats.setdefault("taxonomy_per_100_episodes", {})[stage] = {}
        for yi, (label, sel) in zip(y, conds):
            g = tax[sel & (tax.stage == stage)]
            n = len(g)
            left = 0.0
            row = {"episodes": n}
            for cls in CLASSES:
                k = int((g["class"] == cls).sum())
                rate = 100.0 * k / n
                row[cls] = {"count": k, "per_100": rate}
                if k:
                    ax.barh(yi, rate, left=left, height=0.62, color=COLORS[cls], edgecolor="white", linewidth=1.0)
                    if rate >= 3.5:
                        ax.text(left + rate / 2, yi, f"{rate:.0f}", ha="center", va="center", fontsize=6.5,
                                color="white" if cls in ("filter-blocked", "self-stop") else INK)
                left += rate
            stats["taxonomy_per_100_episodes"][stage][label] = row
        ax.set_title(f"{stage} stage (filter on)", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("timeouts per 100 episodes")
        ax.set_xlim(0, 60)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
    present = [c for c in CLASSES if (tax["class"] == c).any()]   # "other stall" has no cases in these data
    names = {"filter-blocked": "blocked by filter (Type I)", "self-stop": "stopped by policy (Type II)",
             "moving": "still moving", "other stall": "other stall"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in present]
    fig.legend(handles, [names[c] for c in present],
               loc="lower center", ncol=len(present), frameon=False, fontsize=7, bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "stall_taxonomy.pdf")
    plt.close(fig)


def fig_geometry(tax, stats):
    blocked = tax[(tax["class"] == "filter-blocked") & (tax.policy == "ppo")]
    selfstop = tax[(tax["class"] == "self-stop") & (tax.policy == "ppo") & (tax.actions == "mean")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax = axes[0]
    ax.hist(blocked.end_heading_to_obstacle_deg, bins=np.arange(0, 181, 7.5), color=COLORS["filter-blocked"],
            edgecolor="white", linewidth=0.8)
    ax.axvline(90, color=INK2, linestyle="--", linewidth=1)
    ax.text(92, ax.get_ylim()[1] * 0.85, "90°: tangent", fontsize=7, color=INK2)
    ax.set_xlabel("angle between heading and nearest-obstacle direction (deg)")
    ax.set_ylabel("Type I stalls")
    ax.set_title(f"(a) blocked by the filter, all PPO conditions (n = {len(blocked)})", fontsize=8, loc="left")
    ax = axes[1]
    ax.hist(selfstop.end_obstacle_barrier_m, bins=np.arange(0, 1.45, 0.05), color=COLORS["self-stop"],
            edgecolor="white", linewidth=0.8)
    ax.set_xlabel("barrier value h at the stop (m; 0 = keep-out boundary)")
    ax.set_ylabel("Type II stalls")
    ax.set_title(f"(b) stopped by the policy, mean actions (n = {len(selfstop)})", fontsize=8, loc="left")
    for a in axes:
        a.yaxis.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "stall_geometry.pdf")
    plt.close(fig)
    stats["type1_heading_deg"] = {"n": len(blocked), "max": float(blocked.end_heading_to_obstacle_deg.max()),
                                  "median": float(blocked.end_heading_to_obstacle_deg.median()),
                                  "share_below_90": float((blocked.end_heading_to_obstacle_deg < 90).mean()),
                                  "obstacle_barrier_max_m": float(blocked.end_obstacle_barrier_m.max())}
    stats["type2_barrier_m"] = {"n": len(selfstop), "median": float(selfstop.end_obstacle_barrier_m.median()),
                                "q25": float(selfstop.end_obstacle_barrier_m.quantile(0.25)),
                                "q75": float(selfstop.end_obstacle_barrier_m.quantile(0.75)),
                                "min": float(selfstop.end_obstacle_barrier_m.min())}


def indicator_matrix(tax, stage, variant, actions, cls):
    g = tax[(tax.policy == "ppo") & (tax.variant == variant) & (tax.actions == actions) & (tax.stage == stage)]
    g = g.assign(ind=(g["class"] == cls).astype(float))
    return g.pivot_table(index="train_seed", columns="episode_seed", values="ind", aggfunc="mean").to_numpy(float)


def stall_contrasts(tax, stats):
    rows = []
    for stage in ("intermediate", "expert"):
        for label, a, b in [
            ("Type I rate: trained with filter - trained without (mean)", ("filtered", "mean", "filter-blocked"),
             ("unfiltered", "mean", "filter-blocked")),
            ("Type II rate: trained with filter - trained without (mean)", ("filtered", "mean", "self-stop"),
             ("unfiltered", "mean", "self-stop")),
            ("Type II rate, trained with filter: sampled - mean", ("filtered", "sampled", "self-stop"),
             ("filtered", "mean", "self-stop")),
            ("Type I rate, trained with filter: sampled - mean", ("filtered", "sampled", "filter-blocked"),
             ("filtered", "mean", "filter-blocked")),
        ]:
            d, lo, hi = paired_hier_bootstrap(indicator_matrix(tax, stage, *a), indicator_matrix(tax, stage, *b))
            rows.append({"stage": stage, "comparison": label, "difference": d, "ci95_low": lo, "ci95_high": hi})
    stats["stall_contrasts"] = rows
    return pd.DataFrame(rows)


def fig_paired(stall_df, stats):
    succ = pd.read_csv(ROOT / "research" / "audit_outputs" / "paired_comparisons_expert.csv")
    stats["success_contrasts_expert"] = succ.to_dict(orient="records")
    items = [("success: " + r["comparison (expert, success rate)"], r["difference"], r["ci95_low"], r["ci95_high"]) for _, r in succ.iterrows()]
    items += [(f"stall rate: {r.comparison}", r.difference, r.ci95_low, r.ci95_high)
              for r in stall_df.itertuples() if r.stage == "expert"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    y = np.arange(len(items))[::-1]
    for yi, (lab, d, lo, hi) in zip(y, items):
        ax.plot([lo * 100, hi * 100], [yi, yi], color=INK2, linewidth=2, solid_capstyle="round")
        ax.plot(d * 100, yi, "o", markersize=6, color=COLORS["filter-blocked"], markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=6.5)
    ax.set_xlabel("difference in percentage points (expert stage), with 95 % interval")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "paired_differences.pdf")
    plt.close(fig)


def ci_width_fit(stats, reps=3000, seed=0):
    """Approximate scaling of the paired hierarchical-bootstrap interval with seeds and scenarios
    (expert, filter-trained vs unfiltered-trained, sampled actions), fitted on subsamples of the
    stored episodes: var = A/n_seeds + B/n_scen + C/(n_seeds n_scen). Rough: only 5 seeds exist."""
    s = pd.read_csv(ROOT / "results" / "stochastic_eval" / "episodes.csv")
    s = s[(s.stage == "expert") & (s.actions == "sampled")]
    mats = [s[s.variant == v].pivot_table(index="train_seed", columns="episode_seed", values="success",
                                          aggfunc="mean").to_numpy(float) for v in ("filtered", "unfiltered")]
    rng = np.random.default_rng(seed)
    rows = []
    for ns in (2, 3, 5):
        for nc in (25, 50, 100):
            d = []
            for _ in range(reps):
                sc = rng.choice(100, nc, replace=True)
                d.append(mats[0][np.ix_(rng.choice(5, ns), sc)].mean() - mats[1][np.ix_(rng.choice(5, ns), sc)].mean())
            rows.append((ns, nc, float(np.std(d))))
    X = np.array([[1 / ns, 1 / nc, 1 / (ns * nc)] for ns, nc, _ in rows])
    coef = np.clip(np.linalg.lstsq(X, np.array([r[2] ** 2 for r in rows]), rcond=None)[0], 0, None)
    pred = {f"{ns}x{nc}": float(1.96 * np.sqrt(coef @ [1 / ns, 1 / nc, 1 / (ns * nc)]))
            for ns, nc in [(5, 100), (10, 300), (10, 1000), (20, 1000), (10, 3000)]}
    stats["ci_halfwidth_fit"] = {"coef_A_B_C": coef.tolist(), "predicted_95_halfwidth_seeds_x_scenarios": pred}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    tax = pd.read_csv(ROOT / "research" / "audit_outputs" / "stall_taxonomy_all.csv")
    stats = {"source": "research/audit_outputs/stall_taxonomy_all.csv; research/audit_outputs/paired_comparisons_expert.csv"}
    # consistency with the committed benchmark: same outcomes for every PPO-mean and baseline episode
    bench = pd.read_csv(ROOT / "results" / "benchmark" / "episodes.csv")
    bench = bench[bench.safety_filter]
    tm = tax[(tax.actions == "mean")]
    tm = tm.assign(key=np.where(tm.policy == "ppo", tm.variant + "|" + tm.train_seed.astype(str), tm.policy))
    bm = bench.assign(key=np.where(bench.policy_kind == "ppo",
                                   np.where(bench.trained_with_filter == True, "filtered", "unfiltered") + "|"  # noqa: E712
                                   + bench.train_seed.astype(str).str.replace(".0", "", regex=False),
                                   bench.policy_kind))
    tm = tm.assign(key=tm.key.str.replace(".0", "", regex=False))
    merged = tm.merge(bm[["key", "stage", "episode_seed", "outcome"]], on=["key", "stage", "episode_seed"],
                      suffixes=("", "_bench"))
    merged["outcome_bench"] = merged.outcome_bench.replace({"out_of_bounds": "collision"})
    stats["consistency_with_benchmark"] = {"episodes_compared": int(len(merged)),
                                           "outcome_mismatches": int((merged.outcome != merged.outcome_bench).sum())}
    fig_taxonomy(tax, stats)
    fig_geometry(tax, stats)
    stall_df = stall_contrasts(tax, stats)
    fig_paired(stall_df, stats)
    ci_width_fit(stats)
    (GEN / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("consistency_with_benchmark", "type1_heading_deg", "type2_barrier_m")}, indent=2))
    print(stall_df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
