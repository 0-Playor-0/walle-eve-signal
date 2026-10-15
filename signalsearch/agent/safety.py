"""Safety rule: never take an action that could leave too little battery to walk home.

    battery - cost(action) - per_cell * h_after - reserve >= 0

h_after is the upper bound on moves to the nearest known hearth after the action: h + 1 for a move
away (the spec's worst case), h - 1 for a move down the home field, h for anything else.
"""
from __future__ import annotations


def margin(battery: float, cost: float, h_after: float, reserve: float, per_cell: float) -> float:
    return battery - cost - per_cell * h_after - reserve


def is_safe(battery: float, cost: float, h_after: float, reserve: float, per_cell: float) -> bool:
    return margin(battery, cost, h_after, reserve, per_cell) >= -1e-9


def near_failing(battery: float, cost: float, h_after: float, reserve: float, per_cell: float) -> bool:
    """Within 10% of failing: the margin is under a tenth of what the action plus the return needs."""
    need = cost + per_cell * h_after + reserve
    return margin(battery, cost, h_after, reserve, per_cell) < 0.1 * need
