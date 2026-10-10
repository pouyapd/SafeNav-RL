"""
cli.py - command-line interface:  python -m safenav <command> [options]

Commands:
  train         train one PPO agent
  train-many    train several seeds / variants in parallel (one process per run)
  evaluate      evaluate a checkpoint or a baseline on seeded test episodes
  benchmark     evaluate all trained runs and baselines; write CSV/JSON results
  safety-sweep  oracle feasibility and baseline performance versus safety margin
  plot          make the README figures from benchmark, training and sweep outputs

Run `python -m safenav <command> --help` for the options of each command.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from safenav.config import ConfigError, load_config, parse_override


def _load(args) -> dict:
    overrides = dict(parse_override(s) for s in (getattr(args, "set", None) or []))
    return load_config(args.config, overrides=overrides)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="YAML config (default: configs/default_config.yaml)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a config value, e.g. --set ppo.total_timesteps=200000 (repeatable)")


# ── train ────────────────────────────────────────────────────────────────────

def cmd_train(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    config = _load(args)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.timesteps is not None:
        config["ppo"]["total_timesteps"] = args.timesteps
    if args.no_curriculum:
        config.setdefault("curriculum", {})["enabled"] = False
    from safenav.training.ppo_trainer import PPOTrainer
    trainer = PPOTrainer(config, run_dir=args.run_dir, safety_filter=not args.no_safety_filter,
                         verbose=not args.quiet, resume=args.resume)
    trainer.train()
    return 0


def cmd_train_many(args) -> int:
    variants = args.variants
    jobs: List[List[str]] = []
    out = Path(args.out)
    for variant in variants:
        for seed in args.seeds:
            run_dir = out / variant / f"seed_{seed}"
            cmd = [sys.executable, "-m", "safenav", "train", "--seed", str(seed), "--run-dir", str(run_dir),
                   "--quiet", "--threads", "1"]
            if args.config:
                cmd += ["--config", args.config]
            for s in args.set or []:
                cmd += ["--set", s]
            if args.timesteps is not None:
                cmd += ["--timesteps", str(args.timesteps)]
            if variant == "unfiltered":
                cmd.append("--no-safety-filter")
            if args.resume:
                if (run_dir / "train_summary.json").is_file():
                    print(f"[train-many] {run_dir} already finished; skipped", flush=True)
                    continue
                cmd.append("--resume")
            jobs.append(cmd)

    def run(cmd: List[str]) -> int:
        run_dir = Path(cmd[cmd.index("--run-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(run_dir / "stdout.log", "a", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[train-many] {run_dir} finished with code {code} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return code

    print(f"[train-many] {len(jobs)} runs, {args.jobs} in parallel -> {out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run, jobs))
    return 0 if all(c == 0 for c in codes) else 1


# ── evaluate ─────────────────────────────────────────────────────────────────

def cmd_evaluate(args) -> int:
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    from safenav.agents.baselines import GoToGoalPolicy, PotentialFieldPolicy, RandomPolicy
    from safenav.agents.policy import PPOPolicy, load_checkpoint
    from safenav.evaluation.metrics import summarize
    from safenav.evaluation.runner import episode_seeds, evaluate_policy

    if args.checkpoint:
        ckpt_config = load_checkpoint(args.checkpoint).get("config")
        config = _load(args) if (args.config or args.set or not ckpt_config) else ckpt_config
        policy = PPOPolicy.from_checkpoint(args.checkpoint, deterministic=not args.stochastic)
    else:
        config = _load(args)
        policy = {"gotogoal": GoToGoalPolicy(),
                  "potential_field": PotentialFieldPolicy.from_config(config) if args.baseline == "potential_field" else None,
                  "random": RandomPolicy(seed=0)}[args.baseline]

    stages = args.stages or [s["name"] for s in config["curriculum"]["stages"]]
    seed_base = args.seed_base if args.seed_base is not None else config.get("evaluation", {}).get("test_seed", 2_000_000)
    all_results = []
    names = [s["name"] for s in config["curriculum"]["stages"]]
    print(f"Policy: {policy.name} | safety filter at test time: {args.safety_filter}")
    for stage in stages:
        idx = names.index(stage)
        res = evaluate_policy(config, policy, stage, episode_seeds(seed_base, idx, args.episodes),
                              safety_filter=args.safety_filter == "on")
        s = summarize(res)
        print(f"  {stage:13s} success {s['success']['rate']:6.1%} [{s['success']['ci_low']:.0%}, "
              f"{s['success']['ci_high']:.0%}] | collision {s['collision']['rate']:6.1%} | "
              f"timeout {s['timeout']['rate']:6.1%} | SPL {s['spl']:.2f} | "
              f"min clearance (5th pct) {s['min_clearance_p05']:.2f} m")
        all_results += res
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.to_row() for r in all_results]).to_csv(out / "episodes.csv", index=False)
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summarize(all_results), f, indent=2)
        print(f"Saved {out / 'episodes.csv'} and {out / 'summary.json'}")
    return 0


# ── benchmark ────────────────────────────────────────────────────────────────

def cmd_benchmark(args) -> int:
    from safenav.evaluation.benchmark import run_benchmark
    config = _load(args)
    summary = run_benchmark(
        config, Path(args.runs) if args.runs else None, Path(args.out),
        n_episodes=args.episodes, seed_base=args.seed_base, jobs=args.jobs,
        include_baselines=not args.no_baselines, command="python -m safenav " + " ".join(sys.argv[1:]),
    )
    cols = ["method", "safety_filter", "stage", "success_rate", "collision_rate", "timeout_rate", "spl_oracle"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Results written to {args.out}")
    return 0


# ── safety sweep ─────────────────────────────────────────────────────────────

def cmd_safety_sweep(args) -> int:
    from safenav.evaluation.sweep import run_safety_sweep
    config = _load(args)
    df = run_safety_sweep(config, Path(args.out), margins=args.margins, n_episodes=args.episodes, jobs=args.jobs)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


# ── plot ─────────────────────────────────────────────────────────────────────

def cmd_plot(args) -> int:
    from safenav.viz.figures import make_all_figures
    config = _load(args)
    written = make_all_figures(
        config,
        benchmark_dir=Path(args.benchmark) if args.benchmark else None,
        runs_dir=Path(args.runs) if args.runs else None,
        sweep_dir=Path(args.sweep) if args.sweep else None,
        out_dir=Path(args.out),
    )
    for path in written:
        print(f"  wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m safenav", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train one PPO agent")
    _add_config_args(p)
    p.add_argument("--run-dir", default=None, help="output directory (default: training.log_dir)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--no-safety-filter", action="store_true", help="train without the CBF filter")
    p.add_argument("--no-curriculum", action="store_true", help="train directly on the hardest stage")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (default 1)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from <run-dir>/checkpoints/resume.pt (exact)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-many", help="train several seeds/variants in parallel")
    _add_config_args(p)
    p.add_argument("--out", required=True, help="root directory, e.g. runs/benchmark")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="+", default=["filtered", "unfiltered"],
                   choices=["filtered", "unfiltered"])
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--jobs", type=int, default=4, help="parallel processes")
    p.add_argument("--resume", action="store_true", help="skip finished runs and resume interrupted ones")
    p.set_defaults(func=cmd_train_many)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint or baseline")
    _add_config_args(p)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="trained .pt checkpoint")
    src.add_argument("--baseline", choices=["gotogoal", "potential_field", "random"])
    p.add_argument("--stages", nargs="+", default=None, help="difficulty stages (default: all)")
    p.add_argument("--episodes", type=int, default=50, help="episodes per stage")
    p.add_argument("--seed-base", type=int, default=None, help="first episode seed (default: evaluation.test_seed)")
    p.add_argument("--safety-filter", choices=["on", "off"], default="on")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    p.add_argument("--out", default=None, help="directory for episodes.csv and summary.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="evaluate trained runs and baselines")
    _add_config_args(p)
    p.add_argument("--runs", default=None, help="runs root written by train-many")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=None, help="episodes per stage (default: config)")
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--no-baselines", action="store_true")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("safety-sweep", help="feasibility and baseline results vs safety margin")
    _add_config_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=cmd_safety_sweep)

    p = sub.add_parser("plot", help="make figures from outputs")
    _add_config_args(p)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--runs", default=None)
    p.add_argument("--sweep", default=None)
    p.add_argument("--out", default="docs/figures")
    p.set_defaults(func=cmd_plot)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
