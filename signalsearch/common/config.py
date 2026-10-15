"""Config loading, preset inheritance and resolution into a fully explicit, replayable config."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

# Keys under world.sensors that are structural rather than physical and are never perturbed.
_NOT_PERTURBED = {"pos_noise", "persist_block"}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _perturb_tree(tree: dict, rng: np.random.Generator, frac: float) -> dict:
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = _perturb_tree(v, rng, frac)
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or k in _NOT_PERTURBED:
            out[k] = v
        else:
            val = float(v) * float(rng.uniform(1 - frac, 1 + frac))
            if k.startswith("p_"):
                val = min(val, 0.99)
            out[k] = round(val, 6)
    return out


def derive_seeds(seed: int) -> dict:
    names = ["map", "entities", "target", "dogs", "sensors", "signal", "assumed", "agent"]
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {n: int(c.generate_state(1)[0]) for n, c in zip(names, children)}


def resolve_config(cfg: dict, seed: int | None = None) -> dict:
    """Fill seeds and the agent_assumed section. Weather is resolved separately (world.weather)."""
    cfg = copy.deepcopy(cfg)
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg.setdefault("seed", 0)
    if "seeds" not in cfg:
        cfg["seeds"] = derive_seeds(cfg["seed"])
    if not cfg["agent_assumed"].get("resolved"):
        cfg["agent_assumed"] = build_assumed(cfg)
    return cfg


def build_assumed(cfg: dict) -> dict:
    """Agent's beliefs: true specs x U(1-p, 1+p), explicit overrides win. Costs and geometry are exact."""
    w = cfg["world"]
    user = copy.deepcopy(cfg.get("agent_assumed", {}))
    frac = float(user.pop("perturb", 0.10))
    rng = np.random.default_rng(cfg["seeds"]["assumed"])
    t = w["target"]
    drawn = {
        "sensors": _perturb_tree(w["sensors"], rng, frac),
        "temps": _perturb_tree(w["temps"], rng, frac),
        "signal": _perturb_tree(w["signal"], rng, frac),
        "target": _perturb_tree({"p0": t["p0"], "k": t["k"], "drain_fudge": t["drain_fudge"]}, rng, frac),
        "dog_p_move": round(w["dog"]["p_move"] * float(rng.uniform(1 - frac, 1 + frac)), 6),
    }
    out = {
        "resolved": True,
        "perturb": frac,
        "map_size": int(w["map"]["size"]),
        "frame_size": 2 * int(w["map"]["size"]) - 1,
        "costs": copy.deepcopy(w["costs"]),
        "max_steps": int(cfg.get("max_steps", 2000)),
    }
    out = deep_merge(out, drawn)
    out = deep_merge(out, user)  # explicit values in the preset override drawn ones
    return out
