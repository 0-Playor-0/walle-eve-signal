"""BSP castle generator: the interior is recursively partitioned by 1-cell walls, each wall gets door gaps.

Leaves become rooms; a few large leaves are left undivided as courtyards. Arrays are indexed [x, y].
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MapData:
    walls: np.ndarray            # bool [x, y]
    room_id: np.ndarray          # int  [x, y]; -1 for walls and door cells
    rooms: list = field(default_factory=list)       # (x0, y0, x1, y1) inclusive
    courtyards: list = field(default_factory=list)  # indices into rooms
    doors: list = field(default_factory=list)       # door cells (x, y)

    @property
    def size(self) -> int:
        return self.walls.shape[0]

    def free(self, x: int, y: int) -> bool:
        n = self.size
        return 0 <= x < n and 0 <= y < n and not self.walls[x, y]


def bfs_dist(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Exact 4-connected walking distance from start; -1 where unreachable."""
    n0, n1 = walls.shape
    dist = np.full(walls.shape, -1, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        x, y = q.popleft()
        d = dist[x, y] + 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < n0 and 0 <= ny < n1 and dist[nx, ny] < 0 and not walls[nx, ny]:
                dist[nx, ny] = d
                q.append((nx, ny))
    return dist


def is_connected(walls: np.ndarray) -> bool:
    free = np.argwhere(~walls)
    if len(free) == 0:
        return False
    dist = bfs_dist(walls, tuple(free[0]))
    return bool(np.all(dist[~walls] >= 0))


def _split(rect, rng, p, leaves, segments, budget, depth):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0 + 1, y1 - y0 + 1
    mn, mx, cmin = p["min_room"], p["max_room"], p["courtyard_min"]
    # Undivided courtyard
    if budget["courtyards"] > 0 and depth >= 2 and cmin <= w <= 2.2 * cmin and cmin <= h <= 2.2 * cmin \
            and rng.random() < 0.45:
        budget["courtyards"] -= 1
        leaves.append((rect, True))
        return
    can_v = w >= 2 * mn + 1
    can_h = h >= 2 * mn + 1
    small = w <= mx and h <= mx
    if (not can_v and not can_h) or (small and rng.random() < 0.35):
        leaves.append((rect, False))
        return
    if can_v and can_h:
        vertical = w > h if abs(w - h) > 3 else rng.random() < 0.5
    else:
        vertical = can_v
    if vertical:
        s = int(rng.integers(x0 + mn, x1 - mn + 1))
        segments.append(("v", s, y0, y1))
        _split((x0, y0, s - 1, y1), rng, p, leaves, segments, budget, depth + 1)
        _split((s + 1, y0, x1, y1), rng, p, leaves, segments, budget, depth + 1)
    else:
        s = int(rng.integers(y0 + mn, y1 - mn + 1))
        segments.append(("h", s, x0, x1))
        _split((x0, y0, x1, s - 1), rng, p, leaves, segments, budget, depth + 1)
        _split((x0, s + 1, x1, y1), rng, p, leaves, segments, budget, depth + 1)


def _try_generate(p: dict, rng: np.random.Generator) -> MapData | None:
    n = int(p["size"])
    walls = np.zeros((n, n), dtype=bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    lo, hi = p["courtyards"]
    budget = {"courtyards": int(rng.integers(lo, hi + 1))}
    want = budget["courtyards"]
    leaves, segments = [], []
    _split((1, 1, n - 2, n - 2), rng, p, leaves, segments, budget, 0)
    n_court = sum(1 for _, c in leaves if c)
    if n_court < lo or n_court > want:
        return None
    for axis, s, a, b in segments:
        if axis == "v":
            walls[s, a:b + 1] = True
        else:
            walls[a:b + 1, s] = True
    doors = []
    dlo, dhi = p["doors_per_wall"]
    wlo, whi = p["door_width"]
    for axis, s, a, b in segments:
        # door candidates: wall cells with free cells on both perpendicular sides
        cand = []
        for c in range(a, b + 1):
            if axis == "v":
                ok = not walls[s - 1, c] and not walls[s + 1, c]
            else:
                ok = not walls[c, s - 1] and not walls[c, s + 1]
            if ok:
                cand.append(c)
        if not cand:
            return None
        k = int(rng.integers(dlo, dhi + 1)) + (b - a + 1) // int(p.get("door_every", 10 ** 9))
        for c in rng.choice(cand, size=min(k, len(cand)), replace=False):
            width = int(rng.integers(wlo, whi + 1))
            for cc in range(int(c), int(c) + width):
                if cc not in cand:
                    break
                cell = (s, cc) if axis == "v" else (cc, s)
                walls[cell] = False
                doors.append(cell)
    room_id = np.full((n, n), -1, dtype=np.int32)
    rooms, courtyards = [], []
    for i, ((x0, y0, x1, y1), court) in enumerate(leaves):
        rooms.append((x0, y0, x1, y1))
        if court:
            courtyards.append(i)
        room_id[x0:x1 + 1, y0:y1 + 1] = i
    room_id[walls] = -1
    for d in doors:
        room_id[d] = -1
    # sparse rubble inside rooms, never next to a door
    door_set = set(doors)
    free_cells = np.argwhere(room_id >= 0)
    n_rubble = int(len(free_cells) * float(p.get("rubble", 0.0)))
    for idx in rng.choice(len(free_cells), size=n_rubble, replace=False) if n_rubble else []:
        x, y = free_cells[idx]
        # a lone pillar with all 8 neighbours free can never disconnect the map
        if walls[x - 1:x + 2, y - 1:y + 2].any():
            continue
        if any((x + dx, y + dy) in door_set for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        walls[x, y] = True
        room_id[x, y] = -1
    if not is_connected(walls):
        return None
    return MapData(walls=walls, room_id=room_id, rooms=rooms, courtyards=courtyards, doors=doors)


def generate_map(p: dict, rng: np.random.Generator, max_tries: int = 200) -> MapData:
    for _ in range(max_tries):
        m = _try_generate(p, rng)
        if m is not None:
            return m
    raise RuntimeError("map generation failed; check world.map parameters")


def ascii_dump(m: MapData, marks: dict[tuple[int, int], str] | None = None) -> str:
    marks = marks or {}
    rows = []
    for y in range(m.size):
        row = []
        for x in range(m.size):
            if (x, y) in marks:
                row.append(marks[(x, y)])
            elif m.walls[x, y]:
                row.append("#")
            elif m.room_id[x, y] in m.courtyards:
                row.append(",")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)
