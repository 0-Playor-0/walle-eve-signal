"""Run directory writer: trace.jsonl, truth.jsonl, belief/occupancy snapshots, world, report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


class RunLogger:
    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace = open(self.dir / "trace.jsonl", "w")
        self.truth = open(self.dir / "truth.jsonl", "w")
        self.snap: dict[str, np.ndarray] = {}

    def world(self, env) -> None:
        np.savez_compressed(self.dir / "world.npz", walls=env.map.walls, room_id=env.map.room_id)
        with open(self.dir / "world.json", "w") as f:
            f.write(dumps(env.static_truth()))

    def turn(self, obs, decision: dict, agent) -> None:
        rec = {"t": obs.t, "action": decision.get("action"), "reason": decision.get("reason"),
               "cost": obs.cost, "battery": obs.battery, "observation": obs.to_dict()}
        rec.update({k: v for k, v in decision.items() if k not in ("action", "reason")})
        self.trace.write(dumps(rec) + "\n")
        x0, x1, y0, y1 = agent.frame.window
        self.snap[f"b{obs.t:05d}"] = agent.belief.p[x0:x1 + 1, y0:y1 + 1].astype(np.float16)
        self.snap[f"o{obs.t:05d}"] = agent.occ.grid[x0:x1 + 1, y0:y1 + 1].astype(np.int8)
        self.snap[f"w{obs.t:05d}"] = np.array([x0, x1, y0, y1], dtype=np.int16)

    def truth_line(self, snap: dict) -> None:
        self.truth.write(dumps(snap) + "\n")

    def report(self, rep: dict, md: str) -> None:
        with open(self.dir / "report.json", "w") as f:
            json.dump(rep, f, default=_json_default, indent=1)
        (self.dir / "report.md").write_text(md)

    def close(self) -> None:
        self.trace.close()
        self.truth.close()
        np.savez_compressed(self.dir / "beliefs.npz", **self.snap)


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
