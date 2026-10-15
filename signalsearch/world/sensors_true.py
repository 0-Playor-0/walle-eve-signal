"""True sensor models. They read ground truth and return readings in TRUE map coordinates.

Randomness: one sequential stream per env (`rng`), but draws are only made for entities that are in a
sensor's range, so changing something the sensor cannot reach never shifts the noise of anything else.
Thermal noise is partly persistent: a fixed field keyed on (map seed, source cell, observer block).
"""
from __future__ import annotations

import math

import numpy as np

from ..common.actions import DIRS
from ..common.geometry import disk_cells, line_of_sight, walls_crossed
from .entities import Entity
from .mapgen import MapData


def lidar_range(spec: dict, weather: dict) -> float:
    return spec["range"] * max(0.3, 1 - spec["humidity_loss"] * weather["humidity"] - spec["rain_loss"] * weather["rain"])


def lidar(m: MapData, ents: list[Entity], pos, spec: dict, weather: dict, rng) -> dict:
    r = lidar_range(spec, weather)
    cells = disk_cells(pos[0], pos[1], r, m.walls.shape)
    vis = cells[line_of_sight(m.walls, pos, cells)]
    is_wall = m.walls[vis[:, 0], vis[:, 1]]
    visible = {(int(x), int(y)) for x, y in vis[~is_wall]}
    blobs = []
    for e in ents:
        if e.pos not in visible:
            continue
        if rng.random() < spec["p_shape"]:
            shape = e.shape
            if shape in ("humanoid", "small") and rng.random() < spec["shape_confusion"]:
                shape = "small" if shape == "humanoid" else "humanoid"
            blobs.append({"pos": list(e.pos), "shape": shape})
    return {"free": vis[~is_wall].tolist(), "walls": vis[is_wall].tolist(), "blobs": blobs}


def radar(m: MapData, ents: list[Entity], pos, spec: dict, rng) -> list[dict]:
    out = []
    n = spec["pos_noise"]
    for e in ents:
        if not e.heartbeat:
            continue
        d = math.dist(pos, e.pos)
        if d > spec["range"]:
            continue
        w = int(walls_crossed(m.walls, pos, [e.pos])[0])
        if rng.random() < spec["p_detect"] * spec["wall_atten"] ** w:
            dx, dy = (int(v) for v in rng.integers(-n, n + 1, size=2))
            out.append({"pos": [e.x + dx, e.y + dy], "radius": n})
    if rng.random() < spec["false_alarm"]:
        ang, rad = rng.uniform(0, 2 * math.pi), spec["range"] * math.sqrt(rng.random())
        out.append({"pos": [int(round(pos[0] + rad * math.cos(ang))), int(round(pos[1] + rad * math.sin(ang)))],
                    "radius": n})
    return out


def source_temp(e: Entity, temps: dict, ambient: float) -> float:
    return {"target": temps["human"], "dog": temps["dog"], "torch": temps["torch"],
            "hearth": temps["hearth"]}.get(e.kind, ambient)


def thermal(m: MapData, ents: list[Entity], pos, spec: dict, temps: dict, weather: dict, rng, map_seed: int) -> list[dict]:
    out = []
    amb = weather["temperature"]
    att_per_m = spec["humidity_att"] * weather["humidity"] + spec["rain_att"] * weather["rain"]
    pf = spec["persistent_frac"]
    blk = spec["persist_block"]
    for e in ents:
        c = abs(source_temp(e, temps, amb) - amb)
        if c <= 0:
            continue
        d = math.dist(pos, e.pos)
        if d > spec["range"]:
            continue
        w = int(walls_crossed(m.walls, pos, [e.pos])[0])
        s = c / (c + spec["contrast_ref"]) / (1 + (d / spec["d0"]) ** 2) * spec["wall_factor"] ** w * math.exp(-att_per_m * d)
        sigma = spec["blur_base"] + d * spec["blur_per_m"]
        fixed = np.random.default_rng([map_seed, e.x, e.y, pos[0] // blk, pos[1] // blk]).normal(size=3)
        fresh = rng.normal(size=3)
        mix = math.sqrt(pf) * fixed + math.sqrt(1 - pf) * fresh
        s_obs = s + spec["noise_sd"] * mix[0]
        if s_obs < spec["threshold"]:
            continue
        out.append({"pos": [int(round(e.x + sigma * mix[1])), int(round(e.y + sigma * mix[2]))],
                    "radius": int(math.ceil(sigma)), "strength": round(float(min(s_obs, 1.0)), 4)})
    return out


def cv_pattern(pos, facing: str) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    dx, dy = DIRS[facing]
    lx, ly = -dy, dx
    front = (pos[0] + dx, pos[1] + dy)
    far = [(pos[0] + 2 * dx + k * lx, pos[1] + 2 * dy + k * ly) for k in (-1, 0, 1)]
    return front, far


def cv(m: MapData, ents: list[Entity], pos, facing: str, spec: dict, rng) -> dict:
    front, far = cv_pattern(pos, facing)
    n = m.size
    inb = lambda c: 0 <= c[0] < n and 0 <= c[1] < n
    seen = []
    if inb(front):
        seen.append((front, False))
        if not m.walls[front]:
            far_in = [c for c in far if inb(c)]
            for c, ok in zip(far_in, line_of_sight(m.walls, pos, far_in)):
                if ok:
                    seen.append((c, True))
    visible, wallcells, labels = [], [], []
    at = {}
    for e in ents:
        at.setdefault(e.pos, []).append(e)
    for c, is_far in seen:
        if m.walls[c]:
            wallcells.append(list(c))
            continue
        visible.append(list(c))
        for e in at.get(c, []):
            if rng.random() >= spec["p_detect"]:
                continue
            label = e.cv_class
            if e.kind == "armour" and is_far and rng.random() < spec["armour_false_target"]:
                label = "target"
            labels.append({"pos": list(c), "label": label})
    return {"visible": visible, "walls": wallcells, "labels": labels}
