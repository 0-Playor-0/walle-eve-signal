"""Occupancy map in the agent frame: UNKNOWN / FREE / WALL / OUT_OF_BOUNDS, plus known hearths."""
from __future__ import annotations

import numpy as np

UNKNOWN, FREE, WALL, OOB = 0, 1, 2, 3


class Occupancy:
    def __init__(self, size: int):
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.hearths: set[tuple[int, int]] = set()

    def mark(self, cells, value: int) -> int:
        """Mark cells; returns how many changed. OUT_OF_BOUNDS cells stay out of bounds."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        if len(cells) == 0:
            return 0
        cur = self.grid[cells[:, 0], cells[:, 1]]
        ch = (cur != value) & (cur != OOB)
        self.grid[cells[ch, 0], cells[ch, 1]] = value
        return int(ch.sum())

    def apply_window(self, window_mask: np.ndarray) -> None:
        self.grid[~window_mask] = OOB

    @property
    def wall(self) -> np.ndarray:
        return self.grid == WALL

    @property
    def free(self) -> np.ndarray:
        return self.grid == FREE

    @property
    def unknown(self) -> np.ndarray:
        return self.grid == UNKNOWN

    @property
    def walkable(self) -> np.ndarray:
        """Optimistic: unknown counts as walkable."""
        return (self.grid == FREE) | (self.grid == UNKNOWN)

    def hearth_mask(self) -> np.ndarray:
        m = np.zeros(self.grid.shape, dtype=bool)
        for h in self.hearths:
            m[h] = True
        return m


OUTER_RUN = 30  # an interior castle wall this long practically always has a door along it
MAX_GAP = 2     # unknown cells (grazing LIDAR angles) tolerated inside a run of known wall


def _runs(line: np.ndarray, min_len: int):
    """(start, end) inclusive of True runs of at least min_len in a 1-D bool array."""
    if not line.any():
        return []
    d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def _bridge(wall: np.ndarray, unknown: np.ndarray, max_gap: int) -> np.ndarray:
    """1-D: fill unknown gaps of <= max_gap cells that have known wall on both sides."""
    out = wall.copy()
    for a, b in _runs(unknown, 1):
        if b - a + 1 <= max_gap and a > 0 and b < len(wall) - 1 and wall[a - 1] and wall[b + 1]:
            out[a:b + 1] = True
    return out


def outer_wall_mask(occ: "Occupancy", min_len: int = OUTER_RUN, min_free: int = 5) -> np.ndarray:
    """Routing-only heuristic: a known, gapless straight wall run of >= min_len cells with known floor
    on one side and none on the other is probably the outer wall, so the half-plane behind it is not
    used for optimistic routes. Recomputed from scratch, so a free cell seen behind it undoes it."""
    wall, free, unk = occ.wall, occ.free, occ.unknown
    n0, n1 = wall.shape
    out = np.zeros(wall.shape, dtype=bool)
    for x in range(1, n0 - 1):
        if wall[x, :].sum() < min_len - MAX_GAP * 4:
            continue
        for a, b in _runs(_bridge(wall[x, :], unk[x, :], MAX_GAP), min_len):
            east, west = free[x + 1, a:b + 1].sum(), free[x - 1, a:b + 1].sum()
            if east >= min_free and west == 0 and not free[:x, :].any():
                out[:x, :] = True
            elif west >= min_free and east == 0 and not free[x + 1:, :].any():
                out[x + 1:, :] = True
    for y in range(1, n1 - 1):
        if wall[:, y].sum() < min_len - MAX_GAP * 4:
            continue
        for a, b in _runs(_bridge(wall[:, y], unk[:, y], MAX_GAP), min_len):
            south, north = free[a:b + 1, y + 1].sum(), free[a:b + 1, y - 1].sum()
            if south >= min_free and north == 0 and not free[:, :y].any():
                out[:, :y] = True
            elif north >= min_free and south == 0 and not free[:, y + 1:].any():
                out[:, y + 1:] = True
    return out
