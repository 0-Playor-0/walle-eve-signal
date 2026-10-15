"""Pure grid geometry (no world state): bearings, disks, vectorised line-of-sight and wall counting.

Used by the true sensors with the true wall map and by the agent with its own occupancy map.
"""
from __future__ import annotations

import math

import numpy as np


def bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing of a displacement: 0 = N (-y), 90 = E (+x)."""
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_diff(a, b):
    """Signed smallest difference a - b in degrees (works on arrays)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def disk_cells(cx: int, cy: int, r: float, shape: tuple[int, int]) -> np.ndarray:
    """All in-bounds cells within Euclidean radius r of (cx, cy), as an (K, 2) int array."""
    ri = int(math.floor(r))
    xs = np.arange(max(0, cx - ri), min(shape[0], cx + ri + 1))
    ys = np.arange(max(0, cy - ri), min(shape[1], cy + ri + 1))
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    m = (gx - cx) ** 2 + (gy - cy) ** 2 <= r * r
    return np.stack([gx[m], gy[m]], axis=1)


def _segment_samples(origin, targets: np.ndarray):
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    o = np.asarray(origin, dtype=np.float64)
    d = np.abs(targets - o).max(axis=1)
    n = int(math.ceil(3 * (d.max() if len(d) else 0))) + 2
    t = np.linspace(0.0, 1.0, n)
    pts = o[None, None, :] + t[None, :, None] * (targets[:, None, :] - o[None, None, :])
    cells = np.floor(pts + 0.5).astype(np.int64)
    tgt = np.floor(targets + 0.5).astype(np.int64)
    oc = np.floor(o + 0.5).astype(np.int64)
    between = ~(np.all(cells == tgt[:, None, :], axis=2) | np.all(cells == oc[None, None, :], axis=2))
    return cells, between


def _lookup(mask: np.ndarray, cells: np.ndarray) -> np.ndarray:
    x = np.clip(cells[..., 0], 0, mask.shape[0] - 1)
    y = np.clip(cells[..., 1], 0, mask.shape[1] - 1)
    inside = (cells[..., 0] >= 0) & (cells[..., 0] < mask.shape[0]) & (cells[..., 1] >= 0) & (cells[..., 1] < mask.shape[1])
    return mask[x, y] & inside


def line_of_sight(wall_mask: np.ndarray, origin, targets) -> np.ndarray:
    """True where no wall cell lies strictly between origin and target (the target may itself be a wall)."""
    targets = np.asarray(targets).reshape(-1, 2)
    if len(targets) == 0:
        return np.zeros(0, dtype=bool)
    cells, between = _segment_samples(origin, targets)
    hit = _lookup(wall_mask, cells) & between
    return ~hit.any(axis=1)


def walls_crossed(wall_mask: np.ndarray, origin, targets) -> np.ndarray:
    """Number of distinct wall runs crossed strictly between origin and each target."""
    targets = np.asarray(targets).reshape(-1, 2)
    if len(targets) == 0:
        return np.zeros(0, dtype=np.int64)
    cells, between = _segment_samples(origin, targets)
    w = _lookup(wall_mask, cells) & between
    rising = w[:, 1:] & ~w[:, :-1]
    return rising.sum(axis=1) + w[:, 0]


def ray_profile(wall_mask: np.ndarray, unknown_mask: np.ndarray, origin, targets) -> tuple[np.ndarray, np.ndarray]:
    """(known walls crossed, metres of unknown cells) between origin and each target, in one pass."""
    targets = np.asarray(targets).reshape(-1, 2)
    if len(targets) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    cells, between = _segment_samples(origin, targets)
    w = _lookup(wall_mask, cells) & between
    walls = (w[:, 1:] & ~w[:, :-1]).sum(axis=1) + w[:, 0]
    u = _lookup(unknown_mask, cells) & between
    length = np.hypot(*(targets - np.asarray(origin)).T)
    frac = u.sum(axis=1) / np.maximum(1, cells.shape[1])
    return walls, frac * length
