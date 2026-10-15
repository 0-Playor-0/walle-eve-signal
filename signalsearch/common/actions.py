"""Action vocabulary shared by the agent and the environment."""
from __future__ import annotations

from dataclasses import dataclass

# Grid convention everywhere: arrays are indexed [x, y]; y grows downward (south).
DIRS: dict[str, tuple[int, int]] = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_ORDER = ["N", "E", "S", "W"]  # clockwise

MOVE = "move"
TURN = "turn"
LIDAR = "lidar"
RADAR = "radar"
THERMAL = "thermal"
CV = "cv"
CHARGE = "charge"
WAIT = "wait"

SENSORS = (LIDAR, RADAR, THERMAL, CV)
KINDS = (MOVE, TURN, LIDAR, RADAR, THERMAL, CV, CHARGE, WAIT)


@dataclass(frozen=True)
class Action:
    kind: str
    arg: str | None = None  # N/E/S/W for move, L/R for turn

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}")
        if self.kind == MOVE and self.arg not in DIRS:
            raise ValueError(f"move needs a direction, got {self.arg!r}")
        if self.kind == TURN and self.arg not in ("L", "R"):
            raise ValueError(f"turn needs L or R, got {self.arg!r}")

    def __str__(self) -> str:
        return f"{self.kind} {self.arg}" if self.arg else self.kind

    @staticmethod
    def parse(s: str) -> "Action":
        parts = s.split()
        return Action(parts[0], parts[1] if len(parts) > 1 else None)


def turn_steps(facing: str, new: str) -> int:
    """Number of 90-degree turns between two facings (0, 1 or 2)."""
    d = abs(DIR_ORDER.index(facing) - DIR_ORDER.index(new)) % 4
    return min(d, 4 - d)


def rotate(facing: str, side: str) -> str:
    i = DIR_ORDER.index(facing)
    return DIR_ORDER[(i + (1 if side == "R" else -1)) % 4]


def action_cost(action: Action, facing: str, costs: dict) -> float:
    """Battery cost (percent) of an action given the current facing. Charging is a gain, not a cost."""
    if action.kind == MOVE:
        return costs["move"] + costs["turn_per_90"] * turn_steps(facing, action.arg)
    if action.kind == TURN:
        return costs["turn"]
    if action.kind in (LIDAR, RADAR, THERMAL, CV):
        return costs[action.kind]
    return 0.0
