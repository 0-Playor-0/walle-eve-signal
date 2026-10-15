"""Export saved runs for the web replay (docs/index.html), so a browser can play them with no Python.

    python -m signalsearch.viz.web_export samples/clean_rescue samples/decoy_recovery samples/override_recharge

For each run it writes docs/replay/<name>.json (every turn: the agent's cropped map and belief, its plan, scan,
readings and tracks, the truth, the receipt and timeline events) and docs/replay/<name>_castle.png (the true map,
pre-rendered). It also writes docs/replay/sprites.png + sprites.json (procedural pixel sprites) and palette.json
(the viewer's colours), so the web page and the pygame viewer look the same.

Per turn the agent's map is one byte per cell: bits 0-1 = occupancy (0 unknown, 1 free, 2 wall, 3 outside the
map), bit 2 = inside the 90% likely area, bit 3 = inside the 50% likely area; plus one byte of belief glow
(0-255, the same log scale the viewer uses). Both are base64 strings, column-major (x outer, y inner).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from .replay import RunLog  # noqa: E402
from .themes import PALETTE, THEMES  # noqa: E402
from .viewer import Viewer, _likely_threshold, events_of, receipt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SPRITES = ("knight_m", "knight_f", "princess", "prince", "robot", "probe", "dog", "torch", "hearth", "armour")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def _crop(occ_full: np.ndarray, bel_full: np.ndarray, pose, window) -> tuple:
    """The same zoom as the pygame viewer: explored cells, belief mass and the agent."""
    wx0, _, wy0, _ = window
    interest = ((occ_full == 1) | (occ_full == 2)) | (bel_full > bel_full.max() * 0.02)
    idx = np.argwhere(interest)
    px, py = pose[0] - wx0, pose[1] - wy0
    lo = np.minimum(idx.min(axis=0), [px, py]) - 6 if len(idx) else np.array([px - 20, py - 20])
    hi = np.maximum(idx.max(axis=0), [px, py]) + 6 if len(idx) else np.array([px + 20, py + 20])
    side = int(max(40, (hi - lo).max() + 1))
    c = (lo + hi) // 2
    lo = np.clip(c - side // 2, 0, None)
    hi = np.minimum(lo + side, occ_full.shape) - 1
    return (slice(int(lo[0]), int(hi[0]) + 1), slice(int(lo[1]), int(hi[1]) + 1)), wx0 + int(lo[0]), wy0 + int(lo[1])


def _mind(fd: dict) -> dict:
    m = fd["mind"]
    occ_full = np.asarray(m["occ"], dtype=np.int16)
    bel_full = np.asarray(m["belief"], dtype=np.float32)
    sl, x0, y0 = _crop(occ_full, bel_full, m["pose"], m["window"])
    occ, bel = occ_full[sl], bel_full[sl]
    cell = (occ & 3).astype(np.uint8)
    peak = float(bel_full.max())
    glow = np.zeros_like(cell)
    pk = None
    if peak > 0:
        floor = PALETTE["belief_floor"]
        a = np.clip((np.log10(np.maximum(bel / peak, 1e-12)) - math.log10(floor)) / -math.log10(floor), 0, 1)
        glow = np.round(a * 255).astype(np.uint8)
        cell |= (bel >= _likely_threshold(bel_full, 0.9)).astype(np.uint8) << 2
        cell |= (bel >= _likely_threshold(bel_full, 0.5)).astype(np.uint8) << 3
        i = np.unravel_index(int(np.argmax(bel_full)), bel_full.shape)
        wx0, _, wy0, _ = m["window"]
        pk = [wx0 + int(i[0]), wy0 + int(i[1]), round(peak / max(float(bel_full.sum()), 1e-12), 4)]
    w, h = cell.shape
    return {"x0": x0, "y0": y0, "w": w, "h": h, "cells": _b64(cell), "glow": _b64(glow), "peak": pk}


def _readings(r: dict) -> dict:
    out = {}
    if r.get("thermal"):
        out["thermal"] = [[b["pos"][0], b["pos"][1], round(b.get("radius", 1), 1)] for b in r["thermal"]]
    if r.get("radar"):
        out["radar"] = [[b["pos"][0], b["pos"][1]] for b in r["radar"]]
    if "lidar" in r:
        out["lidar"] = [[b["pos"][0], b["pos"][1], b["shape"]] for b in r["lidar"].get("blobs", []) if b["shape"] != "box"]
    if "cv" in r:
        out["cv"] = [x["label"] for x in r["cv"].get("labels", [])]
    return out


def export_run(run_dir: Path, out: Path) -> dict:
    """Write <name>.json and <name>_castle.png for one run; returns its summary."""
    log = RunLog(run_dir)
    name = run_dir.name
    viewer = Viewer(log.cfg, log.walls, log.static, pack="procedural")
    pygame.image.save(viewer._castle, str(out / f"{name}_castle.png"))
    frames = []
    for i, rec in enumerate(log.trace):
        fd = log.frame(i)
        m = fd["mind"]
        tr = fd["truth"]
        life = m.get("life") or {}
        frames.append({
            "t": fd["t"], "bat": round(fd["battery"], 2), "mode": fd.get("mode"), "reason": fd.get("reason"),
            "ov": ({"plans": {k: round(v, 3) for k, v in fd["override"]["plans"].items()},
                    "choice": fd["override"]["choice"]} if fd.get("override") else None),
            "outcome": fd.get("outcome"),
            "pose": m["pose"], "facing": m["facing"], "scan": m.get("scan"), "read": _readings(m.get("readings") or {}),
            "fused": [[round(d["pos"][0], 1), round(d["pos"][1], 1), round(d.get("sigma", 1), 2), d.get("guess", "unknown")]
                      for d in m.get("fused", [])],
            "path": m.get("path"), "goal": m.get("goal"), "hearths": m.get("hearths", []),
            "life": [life.get("mean"), life.get("p10"), life.get("p90")] if life else None,
            "pmove": round(m.get("p_move") or 0.0, 3),
            "mind": _mind(fd),
            "truth": {"r": tr["rescuer"], "f": tr["facing"], "g": tr["target"], "e": round(tr["energy"], 3),
                      "alive": tr["alive"], "rl": round(tr.get("remaining_life", 0.0), 1),
                      "dogs": [[d[0], d[1], d[2]] for d in tr["dogs"]]},
            "rc": receipt(rec["observation"]), "ev": events_of(rec["observation"], rec.get("override")),
        })
    sens = log.cfg["agent_assumed"]["sensors"]
    w = log.static["weather"]
    data = {
        "name": name, "seed": log.cfg.get("seed"), "k": log.cfg["world"]["target"]["k"],
        "theme": log.cfg.get("theme", {}).get("name", "castle"),
        "weather": f"{w.get('temperature')} C, {w.get('humidity')}% RH",
        "location": log.cfg.get("world", {}).get("weather", {}).get("resolved", {}).get("location", ""),
        "n": int(log.walls.shape[0]), "life_turns": log.static["life"]["life_turns"],
        "ranges": {"lidar": sens["lidar"]["range"], "radar": sens["radar"]["range"], "thermal": sens["thermal"]["range"]},
        "entities": [{"id": e["id"], "kind": e["kind"], "pos": e["pos"]} for e in log.static["entities"] if e["kind"] != "target"],
        "tps": float(log.cfg.get("viz", {}).get("turns_per_second", 8)),
        "frames": frames,
    }
    (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    last = frames[-1]
    return {"name": name, "seed": data["seed"], "k": data["k"], "turns": len(frames), "outcome": last["outcome"]}


def export_sprites(out: Path) -> None:
    """Procedural pixel sprites (4 frames each, 16x16) as one sheet, plus the palette."""
    from .assets import Sprites
    sp = Sprites(ROOT / "assets", pack="procedural")
    sheet = pygame.Surface((16 * 4, 16 * len(SPRITES)), pygame.SRCALPHA)
    index = {}
    for row, name in enumerate(SPRITES):
        frames = sp.frames.get(name) or [sp.get(name)]
        for k in range(4):
            f = pygame.transform.scale(frames[k % len(frames)], (16, 16))
            sheet.blit(f, (16 * k, 16 * row))
        index[name] = row
    pygame.image.save(sheet, str(out / "sprites.png"))
    (out / "sprites.json").write_text(json.dumps(index))
    pal = {k: v for k, v in PALETTE.items()}
    (out / "palette.json").write_text(json.dumps({"palette": pal, "themes": THEMES}, default=list))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "docs" / "replay"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pygame.init()
    pygame.display.set_mode((1, 1))
    export_sprites(out)
    index = [export_run(Path(r), out) for r in a.runs]
    (out / "index.json").write_text(json.dumps(index, indent=1))
    for r in index:
        print(r)


if __name__ == "__main__":
    main()
