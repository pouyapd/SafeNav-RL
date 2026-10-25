"""
config.py

Loading, overriding and validating SafeNav-RL configuration files.

All experiment settings live in one YAML file (see configs/default_config.yaml).
`load_config` reads the file, applies optional overrides and validates the
result, so that a typo or an out-of-range value fails early with a clear
message instead of producing a silently wrong experiment.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default_config.yaml"


class ConfigError(ValueError):
    """Raised when a configuration has missing or invalid values."""


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML config, apply overrides and validate it.

    Args:
        path: YAML file (default: configs/default_config.yaml).
        overrides: Nested dict or {"section.key": value} pairs applied on top.

    Returns:
        The validated configuration dictionary.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level.")
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of `cfg` with overrides applied (dotted keys or nested dicts)."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and "." not in key:
            out[key] = _deep_merge(out.get(key, {}), value)
        else:
            set_by_path(out, key, value)
    return out


def _deep_merge(base: Any, update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def set_by_path(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set cfg["a"]["b"] = value for dotted_key "a.b" (creating sections if needed)."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a CLI override such as "safety.enabled=false" into (key, value)."""
    if "=" not in text:
        raise ConfigError(f"Override '{text}' must have the form section.key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override '{text}' has an empty key")
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # YAML 1.1 reads "1e-4" (no decimal point) as a string; accept it as a number.
        try:
            value = float(value)
        except ValueError:
            pass
    return key, value


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = ("env", "reward", "safety", "ppo", "network", "training")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """
    Check a configuration for missing sections and invalid values.

    Raises:
        ConfigError listing every problem found.
    """
    errors: List[str] = []

    for section in _REQUIRED_SECTIONS:
        if not isinstance(cfg.get(section), Mapping):
            errors.append(f"missing section '{section}'")
    if errors:
        raise ConfigError("Invalid config: " + "; ".join(errors))

    def check(section: str, key: str, ok, msg: str, required: bool = True) -> None:
        sec = cfg.get(section) or {}
        if key not in sec:
            if required:
                errors.append(f"{section}.{key} is missing")
            return
        value = sec[key]
        try:
            valid = ok(value)
        except TypeError:
            valid = False
        if not valid:
            errors.append(f"{section}.{key}={value!r}: {msg}")

    positive = lambda v: _is_number(v) and v > 0          # noqa: E731
    non_negative = lambda v: _is_number(v) and v >= 0     # noqa: E731
    positive_int = lambda v: _is_int(v) and v > 0         # noqa: E731

    # env
    check("env", "workspace_size", positive, "must be a positive number (metres)")
    check("env", "dt", positive, "must be a positive number (seconds)")
    check("env", "robot_radius", positive, "must be a positive number (metres)")
    check("env", "goal_radius", positive, "must be a positive number (metres)")
    check("env", "max_steps", positive_int, "must be a positive integer")
    check("env", "sensor_noise_std", non_negative, "must be >= 0")
    check("env", "risk_distance", positive, "must be a positive number (metres)", required=False)

    # sensor (optional section)
    if "sensor" in cfg:
        check("sensor", "num_beams", lambda v: _is_int(v) and v >= 0, "must be an integer >= 0", required=False)
        check("sensor", "max_range", positive, "must be a positive number (metres)", required=False)
        check("sensor", "range_noise_std", non_negative, "must be >= 0", required=False)

    # reward
    for key in ("step_penalty", "distance_shaping_alpha", "risk_penalty_beta",
                "goal_reward", "collision_penalty"):
        check("reward", key, _is_number, "must be a number", required=False)

    # safety
    check("safety", "enabled", lambda v: isinstance(v, bool), "must be true or false")
    check("safety", "safe_distance", non_negative, "must be >= 0 (metres)")
    check("safety", "cbf_gamma", positive, "must be a positive number (1/s)")
    check("safety", "wall_margin", non_negative, "must be >= 0 (metres)", required=False)
    check("safety", "correction_penalty", non_negative, "must be >= 0", required=False)
    check("safety", "qp_solver", lambda v: v == "analytical",
          "only the closed-form 'analytical' solution is implemented", required=False)
    env_dt = cfg["env"].get("dt")
    gamma = cfg["safety"].get("cbf_gamma")
    if _is_number(env_dt) and _is_number(gamma) and env_dt > 0 and gamma > 0 and gamma * env_dt > 1.0:
        errors.append(
            f"safety.cbf_gamma * env.dt = {gamma * env_dt:.3f} must be <= 1 "
            "(otherwise the discrete-time barrier condition is not meaningful)"
        )

    # curriculum (optional)
    if "curriculum" in cfg:
        cur = cfg["curriculum"] or {}
        stages = cur.get("stages", [])
        if cur.get("enabled", True) and not stages:
            errors.append("curriculum.stages must contain at least one stage when curriculum is enabled")
        check("curriculum", "advance_threshold", lambda v: _is_number(v) and 0 < v <= 1,
              "must be in (0, 1]", required=False)
        check("curriculum", "window_size", positive_int, "must be a positive integer", required=False)
        errors.extend(_validate_stages(stages))

    # ppo
    check("ppo", "total_timesteps", positive_int, "must be a positive integer")
    check("ppo", "rollout_steps", positive_int, "must be a positive integer")
    check("ppo", "n_epochs", positive_int, "must be a positive integer")
    check("ppo", "batch_size", positive_int, "must be a positive integer")
    check("ppo", "gamma", lambda v: _is_number(v) and 0 < v <= 1, "must be in (0, 1]")
    check("ppo", "gae_lambda", lambda v: _is_number(v) and 0 <= v <= 1, "must be in [0, 1]")
    check("ppo", "clip_epsilon", positive, "must be positive")
    check("ppo", "learning_rate", positive, "must be positive")
    check("ppo", "lr_schedule", lambda v: v in ("constant", "linear"), "must be 'constant' or 'linear'",
          required=False)
    check("ppo", "target_kl", lambda v: v is None or positive(v), "must be null or positive", required=False)
    ppo = cfg["ppo"]
    if _is_int(ppo.get("batch_size")) and _is_int(ppo.get("rollout_steps")) \
            and ppo["batch_size"] > ppo["rollout_steps"]:
        errors.append("ppo.batch_size must not exceed ppo.rollout_steps")

    # network
    check("network", "hidden_dims",
          lambda v: isinstance(v, list) and len(v) > 0 and all(positive_int(h) for h in v),
          "must be a non-empty list of positive integers")
    check("network", "activation", lambda v: v in ("tanh", "relu"), "must be 'tanh' or 'relu'",
          required=False)
    check("network", "action_dim", lambda v: v == 2, "must be 2 (v, omega)", required=False)
    net = cfg["network"]
    if _is_number(net.get("log_std_min")) and _is_number(net.get("log_std_max")) \
            and net["log_std_min"] >= net["log_std_max"]:
        errors.append("network.log_std_min must be smaller than network.log_std_max")

    # training
    check("training", "seed", _is_int, "must be an integer")
    check("training", "device", lambda v: v in ("auto", "cpu", "cuda"), "must be 'auto', 'cpu' or 'cuda'",
          required=False)
    check("training", "eval_every", positive_int, "must be a positive integer", required=False)
    check("training", "eval_episodes", lambda v: _is_int(v) and v >= 0, "must be an integer >= 0",
          required=False)
    check("training", "resume_every", lambda v: _is_int(v) and v >= 0, "must be an integer >= 0 (0 = off)",
          required=False)

    if errors:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))


def _validate_stages(stages: Iterable[Any]) -> List[str]:
    errors = []
    for i, stage in enumerate(stages or []):
        where = f"curriculum.stages[{i}]"
        if not isinstance(stage, Mapping):
            errors.append(f"{where} must be a mapping")
            continue
        missing = [k for k in ("min_obstacles", "max_obstacles", "min_goal_dist", "max_goal_dist", "max_steps")
                   if k not in stage]
        if missing:
            errors.append(f"{where} is missing {', '.join(missing)}")
            continue
        if not (_is_int(stage["min_obstacles"]) and _is_int(stage["max_obstacles"])
                and 0 <= stage["min_obstacles"] <= stage["max_obstacles"]):
            errors.append(f"{where}: need integers 0 <= min_obstacles <= max_obstacles")
        if not (_is_number(stage["min_goal_dist"]) and _is_number(stage["max_goal_dist"])
                and 0 < stage["min_goal_dist"] <= stage["max_goal_dist"]):
            errors.append(f"{where}: need 0 < min_goal_dist <= max_goal_dist")
        if not (_is_int(stage["max_steps"]) and stage["max_steps"] > 0):
            errors.append(f"{where}: max_steps must be a positive integer")
    return errors


def get_stage(cfg: Mapping[str, Any], name_or_index: str | int) -> Dict[str, Any]:
    """Return a curriculum stage by name ("beginner") or index (0)."""
    stages = (cfg.get("curriculum") or {}).get("stages") or []
    if isinstance(name_or_index, int):
        if not 0 <= name_or_index < len(stages):
            raise ConfigError(f"Stage index {name_or_index} out of range (0..{len(stages) - 1})")
        return dict(stages[name_or_index])
    for stage in stages:
        if stage.get("name") == name_or_index:
            return dict(stage)
    names = [s.get("name") for s in stages]
    raise ConfigError(f"Unknown stage '{name_or_index}'. Available: {names}")
