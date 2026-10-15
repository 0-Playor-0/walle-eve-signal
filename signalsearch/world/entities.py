"""Static and moving entities in the castle (ground truth)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapgen import MapData

# kind -> (lidar shape, cv class, has heartbeat)
SIGNATURES = {
    "target": ("humanoid", "target", True),
    "dog": ("small", "dog", True),
    "torch": ("small", "fire", False),
    "hearth": ("box", "hearth", False),
    "armour": ("humanoid", "armour", False),
}


@dataclass
class Entity:
    id: int
    kind: str
    x: int
    y: int

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def shape(self) -> str:
        return SIGNATURES[self.kind][0]

    @property
    def cv_class(self) -> str:
        return SIGNATURES[self.kind][1]

    @property
    def heartbeat(self) -> bool:
        return SIGNATURES[self.kind][2]


def _free_neighbours(m: MapData, x: int, y: int) -> list[tuple[int, int]]:
    return [(x + dx, y + dy) for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)) if m.free(x + dx, y + dy)]


def random_step(m: MapData, rng: np.random.Generator, x: int, y: int, room_bias: float = 1.0) -> tuple[int, int]:
    nb = _free_neighbours(m, x, y)
    if not nb:
        return (x, y)
    here = m.room_id[x, y]
    w = np.array([room_bias if (here >= 0 and m.room_id[c] == here) else 1.0 for c in nb])
    return nb[int(rng.choice(len(nb), p=w / w.sum()))]


def place_entities(m: MapData, rng: np.random.Generator, counts: dict, rescuer: tuple[int, int],
                   target: tuple[int, int]) -> list[Entity]:
    """Hearth at the rescuer's spawn, then the rest spread out; never two entities on one cell."""
    taken = {rescuer, target}
    room_cells = np.argwhere(m.room_id >= 0)
    near_wall = np.array([c for c in room_cells
                          if m.walls[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2].any()])
    ents: list[Entity] = [Entity(0, "hearth", *rescuer)]

    def pick(pool, min_sep_from=None, min_sep=0):
        for _ in range(500):
            x, y = (int(v) for v in pool[rng.integers(len(pool))])
            if (x, y) in taken:
                continue
            if min_sep_from and any(abs(x - a) + abs(y - b) < min_sep for a, b in min_sep_from):
                continue
            taken.add((x, y))
            return x, y
        raise RuntimeError("could not place entity")

    def n(key):
        lo, hi = counts[key]
        return int(rng.integers(lo, hi + 1))

    hearths = [rescuer]
    for _ in range(n("hearths") - 1):
        pos = pick(room_cells, hearths, 25)
        hearths.append(pos)
        ents.append(Entity(len(ents), "hearth", *pos))
    for _ in range(n("torches")):
        ents.append(Entity(len(ents), "torch", *pick(near_wall, [e.pos for e in ents if e.kind == "torch"], 10)))
    for _ in range(n("armour")):
        ents.append(Entity(len(ents), "armour", *pick(near_wall)))
    for _ in range(n("dogs")):
        ents.append(Entity(len(ents), "dog", *pick(room_cells, [target, rescuer], 10)))
    return ents
