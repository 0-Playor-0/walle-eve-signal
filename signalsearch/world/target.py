"""The target's true dynamics and life budget."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .entities import random_step
from .mapgen import MapData


def life_budget(d_path: int, k: float) -> dict:
    """Per-map life so missions are fair (see spec section 3)."""
    est_battery_use = 0.2 * 1.8 * d_path + 60
    charging_allowance = 15 * max(0, math.ceil(est_battery_use / 100) - 1)
    t_expected = 1.5 * 1.2 * d_path + 10 + charging_allowance
    return {"d_path": int(d_path), "est_battery_use": est_battery_use, "charging_allowance": charging_allowance,
            "t_expected": t_expected, "life_turns": int(round(k * t_expected))}


@dataclass
class Target:
    x: int
    y: int
    p0: float
    d: float
    room_bias: float
    energy: float = 1.0
    alive: bool = True
    moved_last: bool = False

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def p_move(self) -> float:
        return self.p0 * self.energy

    def step(self, m: MapData, rng: np.random.Generator, forced: tuple[int, int] | None = None,
             controlled: bool = False) -> None:
        """One turn. `controlled` = a human chooses (`forced` neighbour or None to stay)."""
        if not self.alive:
            return
        if controlled:
            move_to = forced if forced is not None and m.free(*forced) and \
                abs(forced[0] - self.x) + abs(forced[1] - self.y) == 1 else None
        else:
            move_to = random_step(m, rng, self.x, self.y, self.room_bias) if rng.random() < self.p_move else None
        self.moved_last = move_to is not None and move_to != self.pos
        if self.moved_last:
            self.x, self.y = move_to
        self.energy -= self.d * (2 if self.moved_last else 1)
        if self.energy <= 0:
            self.energy = 0.0
            self.alive = False

    def remaining_life_estimate(self) -> float:
        """True expected turns left at the current energy (for the truth display only)."""
        per_turn = self.d * (1 + self.p_move)
        return self.energy / per_turn if per_turn > 0 else float("inf")
