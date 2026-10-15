"""Distance fields (vectorised wavefront BFS) instead of per-query A*.

- exploration field: from the agent's cell, unknown treated as free (optimistic)
- home field: multi-source from known hearths through known-free cells only (pessimistic, always valid)
"""
from __future__ import annotations

import numpy as np

from ..common.actions import DIRS


def bfs_field(passable: np.ndarray, sources: np.ndarray, max_iter: int = 100000) -> np.ndarray:
    """Distance (int32) from any source through passable cells; -1 where unreachable."""
    dist = np.full(passable.shape, -1, dtype=np.int32)
    frontier = sources & passable
    if not frontier.any():
        return dist
    dist[frontier] = 0
    visited = frontier.copy()
    d = 0
    while frontier.any() and d < max_iter:
        d += 1
        nb = np.zeros_like(frontier)
        nb[1:, :] |= frontier[:-1, :]
        nb[:-1, :] |= frontier[1:, :]
        nb[:, 1:] |= frontier[:, :-1]
        nb[:, :-1] |= frontier[:, 1:]
        new = nb & passable & ~visited
        dist[new] = d
        visited |= new
        frontier = new
    return dist


def path_to(field: np.ndarray, goal) -> list[tuple[int, int]] | None:
    """Cells from the field's source to goal (excluding the source), following decreasing distance."""
    gx, gy = int(goal[0]), int(goal[1])
    if field[gx, gy] < 0:
        return None
    path = [(gx, gy)]
    x, y = gx, gy
    while field[x, y] > 0:
        d = field[x, y]
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == d - 1:
                x, y = nx, ny
                break
        else:
            return None
        path.append((x, y))
    path.pop()  # the source itself
    return path[::-1]


def step_down(field: np.ndarray, pos) -> tuple[int, int] | None:
    """Neighbour that is one step closer to the field's sources."""
    x, y = pos
    d = field[x, y]
    if d <= 0:
        return None
    for dx, dy in DIRS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and 0 <= field[nx, ny] < d:
            return (nx, ny)
    return None


def direction(a, b) -> str:
    d = (b[0] - a[0], b[1] - a[1])
    for k, v in DIRS.items():
        if v == d:
            return k
    raise ValueError(f"{a} -> {b} is not a single step")


def cost_field(cost: np.ndarray, sources: np.ndarray, max_iter: int = 4000) -> np.ndarray:
    """Weighted shortest-path cost from the sources (vectorised Bellman-Ford relaxation).
    `cost` is the price of entering each cell (np.inf = impassable). Returns np.inf where unreachable."""
    dist = np.full(cost.shape, np.inf)
    dist[sources & np.isfinite(cost)] = 0.0
    for _ in range(max_iter):
        nb = np.full(cost.shape, np.inf)
        nb[1:, :] = np.minimum(nb[1:, :], dist[:-1, :])
        nb[:-1, :] = np.minimum(nb[:-1, :], dist[1:, :])
        nb[:, 1:] = np.minimum(nb[:, 1:], dist[:, :-1])
        nb[:, :-1] = np.minimum(nb[:, :-1], dist[:, 1:])
        new = np.minimum(dist, nb + cost)
        if np.array_equal(new, dist):
            break
        dist = new
    return dist


def path_down(field: np.ndarray, goal) -> list[tuple[int, int]] | None:
    """Like path_to, for a float cost field: from goal step to the cheapest neighbour until the source."""
    x, y = int(goal[0]), int(goal[1])
    if not np.isfinite(field[x, y]):
        return None
    path = [(x, y)]
    for _ in range(field.size):
        if field[x, y] <= 0:
            break
        best = None
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] < field[x, y] and \
                    (best is None or field[nx, ny] < field[best]):
                best = (nx, ny)
        if best is None:
            return None
        x, y = best
        path.append(best)
    path.pop()
    return path[::-1]
