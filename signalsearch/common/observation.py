"""The only thing that crosses from the world to the agent: a JSON-serialisable Observation."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field


@dataclass
class Observation:
    t: int
    action: str
    cost: float
    battery: float
    pose: list[int]          # agent frame
    facing: str
    blocked: bool = False
    on_hearth: bool = False
    charged: float = 0.0
    readings: dict = field(default_factory=dict)
    episode: dict = field(default_factory=lambda: {"done": False, "reason": None})
    initial_signal: dict | None = None   # t=0 only
    weather: dict | None = None          # t=0 only

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["initial_signal"] is None:
            del d["initial_signal"]
        if d["weather"] is None:
            del d["weather"]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        d = copy.deepcopy(d)
        return Observation(**d)

    @property
    def done(self) -> bool:
        return bool(self.episode.get("done"))
