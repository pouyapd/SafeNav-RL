"""
verify_manuscript_round4.py — check that every round-4 number quoted in the manuscript matches the saved results.

    python research/scripts/verify_manuscript_round4.py

Reads research/experiments/round4/results/summary.json and timeouts/summary.json, formats each
quantity as the manuscript does, and checks that the string occurs in paper_draft/manuscript.tex
(and, for the supplement, paper_draft/supplementary.tex if present). Exit code 1 on any miss.
Round-3 numbers are checked by research/experiments/round4/audit_round3.py (section C).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R4 = ROOT / "research" / "experiments" / "round4"


def pct(x):
    return f"{100 * x:.1f}"


def sgn(x):
    return f"$-{abs(100 * x):.1f}$" if x < 0 else f"$+{100 * x:.1f}$"


def ci_tab(c, neg):
    lo, hi = 100 * c[0], 100 * c[1]
    if neg:
        return f"[$-{abs(lo):.1f}$, $-{abs(hi):.1f}$]"
    return f"[{lo:.1f}, {hi:.1f}]"


def main():
    s = json.loads((R4 / "results" / "summary.json").read_text())
    tdir = json.loads((R4 / "timeouts" / "summary.json").read_text())
    tex = (ROOT / "paper_draft" / "manuscript.tex").read_text(encoding="utf-8")
    sup_p = ROOT / "paper_draft" / "supplementary.tex"
    sup = sup_p.read_text(encoding="utf-8") if sup_p.is_file() else ""
    checks = []

    def need(label, text, where="ms"):
        hay = tex if where == "ms" else sup
        checks.append((label, text, text in hay, where))

    # Part P
    P = s["P"]
    need("P fast call p99", f"{P['fast']['call_wall_ms']['p99']:.1f}\\,ms")
    need("P reference call p99", f"{P['reference']['call_wall_ms']['p99']:.1f}\\,ms")
    need("P fast steps >= 100 ms", f"{P['fast']['steps_over_100ms']} of 419{{,}}777")
    cr = P["cpu_ratio_fast_over_reference_per_episode"]
    need("P CPU ratio", f"{cr['mean']:.2f} [{cr['ci95'][0]:.2f}, {cr['ci95'][1]:.2f}]")
    need("P max CPU of a fast call", f"{P['fast']['call_cpu_ms']['max']:.1f}\\,ms")
    need("P identical episodes", f"{P['identical_episodes']['identical']:,}".replace(",", "{,}") + " of "
         + f"{P['identical_episodes']['matched']:,}".replace(",", "{,}"))

    # Part R, Table 7
    names = {"localisation_error": "position noise", "map_error": "map error", "sensed_map": "sensed map",
             "intermediate": "intermediate", "dense": "dense", "heading": "random heading"}
    rates = s["R"]["rates"]
    cmeth = {"localisation_error": ("expert", "C_loc"), "map_error": ("expert", "C_map"), "sensed_map": ("expert", "C_sensed"),
             "intermediate": ("intermediate", "C"), "dense": ("dense", "C"), "heading": ("heading", "C")}
    for k in names:
        st, m = cmeth[k]
        for agent, tag in (("filtered", "with_filter"), ("unfiltered", "without_filter")):
            b = rates[st][f"baseline|fast|{agent}"]
            c = rates[st][f"{m}|fast|{agent}"]
            need(f"R {k} {agent} deadlock", f"{pct(b['deadlock']['mean'])} $\\to$ {pct(c['deadlock']['mean'])}")
            need(f"R {k} {agent} success", f"{pct(b['success']['mean'])} $\\to$ {pct(c['success']['mean'])}")
            d = s["R"]["primary"][f"{k}|deadlock|{tag}"]
            need(f"R {k} {agent} d deadlock", f"{sgn(d['difference'])} {ci_tab(d['ci95'], True)}")
            u = s["R"]["primary"][f"{k}|success|{tag}"]
            need(f"R {k} {agent} d success", f"{sgn(u['difference'])} {ci_tab(u['ci95'], False)}")
            assert d["p_holm"] < 1e-4 and u["p_holm"] < 1e-4
    for tag in ("with_filter", "without_filter"):
        g = s["R"]["degradation_unadjusted"][f"C_map_minus_C|deadlock|{tag}"]
        need(f"map degradation {tag}", f"{pct(g['difference'])} points [{pct(g['ci95'][0])}, {pct(g['ci95'][1])}]")

    # Part F
    F = s["F"]["rates"]
    need("F planner only success", f"{pct(F['planner_only|fast|map']['success']['mean'])}\\,\\%")
    for agent in ("filtered", "unfiltered"):
        need(f"F C success {agent}", f"{pct(F[f'C|fast|{agent}']['success']['mean'])}\\,\\%")
        need(f"F nofilter collision {agent}", f"{pct(F[f'nofilter|fast|{agent}']['collision']['mean'])}\\,\\%")
        need(f"F nofilter success {agent}", f"{pct(F[f'nofilter|fast|{agent}']['success']['mean'])}\\,\\%")
    for tag in ("with_filter", "without_filter"):
        c = s["F"]["comparisons_unadjusted"][f"C_minus_planner_only|success|{tag}"]
        need(f"F C-planner {tag}", f"{abs(100 * c['difference']):.1f} points [{abs(100 * c['ci95'][1]):.1f}, {abs(100 * c['ci95'][0]):.1f}]")
    need("F path follower", f"{pct(F['path_follower|fast|map']['success']['mean'])}\\,\\%")
    need("F path follower filter-blocked", f"{pct(F['path_follower|fast|map']['filter_blocked']['mean'])}\\,\\%")
    need("F inflated follower", f"{pct(F['path_follower_inflated|fast|map']['success']['mean'])}\\,\\%")
    need("F planner deadlock", f"{pct(F['planner_only|fast|map']['deadlock']['mean'])}\\,\\%")
    need("F planner moving", f"{pct(F['planner_only|fast|map']['moving']['mean'])}\\,\\%")

    # Part T
    dm = tdir["decomposition_median"]
    need("T speed", f"{tdir['decomposition_mean']['mean_exec_speed']:.2f}\\,m/s")
    need("T filter share", f"{100 * dm['filter_limited_share']:.0f}\\,\\%")
    need("T start geodesic", f"{dm['start_geo_m']:.1f}\\,m")
    need("T policy regress", f"{100 * tdir['share_policy_regress_over_1m']:.0f}\\,\\%")
    cf = tdir["counterfactuals"]
    for v, lab in (("max_steps_800", "a limit of 800 steps"), ("planner_only", "the planner alone from the start"),
                   ("window_15", "a trigger window of 1.5\\,s instead of 3\\,s"), ("no_handback", "no hand-back to the policy")):
        need(f"T {v}", f"{lab}: {100 * cf[v]['success']:.0f}\\,\\%")
    assert tdir["reproduced_moving_timeouts"] == 700

    bad = [c for c in checks if not c[2]]
    for lab, txt, ok, where in checks:
        print(("OK   " if ok else "MISS ") + f"{lab:38s} {txt}")
    print(f"\n{len(checks)} checks, {len(bad)} missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
