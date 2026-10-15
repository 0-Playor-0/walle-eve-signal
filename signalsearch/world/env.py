"""The environment: owns all ground truth, turns Actions into Observations (agent frame)."""
from __future__ import annotations

import math

import numpy as np

from ..common.actions import CHARGE, CV, DIRS, LIDAR, MOVE, RADAR, THERMAL, TURN, Action, action_cost, rotate
from ..common.geometry import bearing_deg
from ..common.observation import Observation
from . import sensors_true as S
from .entities import Entity, place_entities, random_step
from .mapgen import MapData, bfs_dist, generate_map
from .target import Target, life_budget

RESCUE_CV_WINDOW = 10


class Env:
    def __init__(self, cfg: dict, extra_entities: list[tuple[str, int, int]] | None = None):
        """cfg must be resolved (seeds + weather). `extra_entities` is a test hook to add unseen decoys."""
        self.cfg = cfg
        w = cfg["world"]
        self.w = w
        seeds = cfg["seeds"]
        self.map_seed = seeds["map"]
        self.weather = {k: w["weather"]["resolved"][k] for k in ("temperature", "humidity", "rain")}
        self.map: MapData = generate_map(w["map"], np.random.default_rng(seeds["map"]))
        self.rng_target = np.random.default_rng(seeds["target"])
        self.rng_dogs = np.random.default_rng(seeds["dogs"])
        self.rng_sensors = np.random.default_rng(seeds["sensors"])
        self._spawn(np.random.default_rng(seeds["entities"]), np.random.default_rng(seeds["signal"]))
        for kind, x, y in extra_entities or []:
            self.entities.append(Entity(len(self.entities), kind, x, y))
        self.max_steps = int(cfg.get("max_steps", 2000))
        self.costs = w["costs"]
        self.t = 0
        self.battery = float(self.costs["battery_start"])
        self.facing = "N"
        self.pos = self.spawn
        self.last_cv_target_t = -10 ** 9
        self.done = False
        self.reason = None

    # ---- construction ---------------------------------------------------------------------------
    def _spawn(self, rng: np.random.Generator, rng_sig: np.random.Generator) -> None:
        m, w = self.map, self.w
        room_cells = np.argwhere(m.room_id >= 0)
        sp = w["spawn"]
        sig = w["signal"]
        tp = w["target"]
        for _ in range(10000):
            rescuer = tuple(int(v) for v in room_cells[rng.integers(len(room_cells))])
            start = tuple(int(v) for v in room_cells[rng.integers(len(room_cells))])
            age = int(rng_sig.integers(0, sig["max_age"] + 1))
            pos = start
            walk = np.random.default_rng(rng.integers(2 ** 32))
            for _ in range(age):  # pre-roll the target's wander since the distress call (no drain)
                if walk.random() < tp["p0"]:
                    pos = random_step(m, walk, *pos, tp["room_bias"])
            md = abs(pos[0] - rescuer[0]) + abs(pos[1] - rescuer[1])
            if sp["min_dist"] <= md <= sp["max_dist"] and pos != rescuer:
                break
        else:
            raise RuntimeError("could not satisfy spawn distance")
        self.spawn = rescuer
        dist = bfs_dist(m.walls, rescuer)
        self.life = life_budget(int(dist[pos]), tp["k"])
        d = tp["drain_multiplier"] / (tp["drain_fudge"] * self.life["life_turns"])
        self.target = Target(pos[0], pos[1], tp["p0"], d, tp["room_bias"])
        self.entities = place_entities(m, rng, w["entities"], rescuer, pos)
        self.target_entity = Entity(len(self.entities), "target", *pos)
        self.entities.insert(0, self.target_entity)
        # one-time distress call from where the target was `age` turns ago
        dx, dy = start[0] - rescuer[0], start[1] - rescuer[1]
        true_bearing = bearing_deg(dx, dy)
        dd = math.hypot(dx, dy)
        noise_b = rng_sig.uniform(-sig["bearing_noise_deg"], sig["bearing_noise_deg"])
        strength = 1.0 / (1.0 + (dd / sig["d0"]) ** 2) * math.exp(rng_sig.normal(0, sig["strength_noise"]))
        self.signal_truth = {"emitted_from": list(start), "true_bearing": true_bearing, "true_distance": dd}
        self.initial_signal = {"bearing_deg": round((true_bearing + noise_b) % 360, 2),
                               "strength": round(float(min(strength, 1.0)), 4), "age": age}

    # ---- frame conversion -----------------------------------------------------------------------
    @property
    def offset(self) -> tuple[int, int]:
        """agent_frame = true + offset."""
        c = self.map.size - 1
        return (c - self.spawn[0], c - self.spawn[1])

    def to_agent(self, p) -> list[int]:
        ox, oy = self.offset
        return [int(p[0]) + ox, int(p[1]) + oy]

    def _conv(self, items: list[dict]) -> list[dict]:
        return [{**it, "pos": self.to_agent(it["pos"])} for it in items]

    # ---- stepping -------------------------------------------------------------------------------
    @property
    def on_hearth(self) -> bool:
        return any(e.kind == "hearth" and e.pos == self.pos for e in self.entities)

    def reset(self) -> Observation:
        return Observation(t=0, action="spawn", cost=0.0, battery=self.battery, pose=self.to_agent(self.pos),
                           facing=self.facing, on_hearth=self.on_hearth,
                           initial_signal=dict(self.initial_signal), weather=dict(self.weather))

    def step(self, action: Action, target_move: tuple[int, int] | None = None,
             target_controlled: bool = False) -> Observation:
        if self.done:
            raise RuntimeError("episode is over")
        self.t += 1
        cost = action_cost(action, self.facing, self.costs)
        readings: dict = {}
        blocked = False
        charged = 0.0
        if cost > self.battery + 1e-9:
            self._finish("battery_depleted")
            return self._obs(action, 0.0, readings, blocked, charged)
        self.battery = round(self.battery - cost, 6)
        k = action.kind
        if k == MOVE:
            self.facing = action.arg
            dx, dy = DIRS[action.arg]
            nxt = (self.pos[0] + dx, self.pos[1] + dy)
            if self.map.free(*nxt):
                self.pos = nxt
            else:
                blocked = True
        elif k == TURN:
            self.facing = rotate(self.facing, action.arg)
        elif k == LIDAR:
            r = S.lidar(self.map, self.entities, self.pos, self.w["sensors"]["lidar"], self.weather, self.rng_sensors)
            readings["lidar"] = {"free": [self.to_agent(c) for c in r["free"]],
                                 "walls": [self.to_agent(c) for c in r["walls"]], "blobs": self._conv(r["blobs"])}
        elif k == RADAR:
            readings["radar"] = self._conv(S.radar(self.map, self.entities, self.pos, self.w["sensors"]["radar"],
                                                   self.rng_sensors))
        elif k == THERMAL:
            readings["thermal"] = self._conv(S.thermal(self.map, self.entities, self.pos, self.w["sensors"]["thermal"],
                                                       self.w["temps"], self.weather, self.rng_sensors, self.map_seed))
        elif k == CV:
            r = S.cv(self.map, self.entities, self.pos, self.facing, self.w["sensors"]["cv"], self.rng_sensors)
            if any(lab["label"] == "target" for lab in r["labels"]):
                self.last_cv_target_t = self.t
            readings["cv"] = {"visible": [self.to_agent(c) for c in r["visible"]],
                              "walls": [self.to_agent(c) for c in r["walls"]], "labels": self._conv(r["labels"])}
        elif k == CHARGE and self.on_hearth:  # one turn on a hearth refills to full
            charged = round(100.0 - self.battery, 6)
            self.battery = 100.0
        self._check_rescue()
        if not self.done:
            self._world_tick(target_move, target_controlled)
            self._check_rescue()
        if not self.done:
            if not self.target.alive:
                self._finish("target_died")
            elif self.battery <= 1e-9 and not self.on_hearth:
                self._finish("battery_depleted")
            elif self.t >= self.max_steps:
                self._finish("step_cap")
        return self._obs(action, cost, readings, blocked, charged)

    def _world_tick(self, target_move, target_controlled) -> None:
        self.target.step(self.map, self.rng_target, target_move, target_controlled)
        self.target_entity.x, self.target_entity.y = self.target.pos
        p = self.w["dog"]["p_move"]
        for e in self.entities:
            if e.kind == "dog" and self.rng_dogs.random() < p:
                e.x, e.y = random_step(self.map, self.rng_dogs, e.x, e.y)

    def _check_rescue(self) -> None:
        if self.target.alive and self.pos == self.target.pos and self.t - self.last_cv_target_t <= RESCUE_CV_WINDOW:
            self._finish("rescued")

    def _finish(self, reason: str) -> None:
        self.done, self.reason = True, reason

    def _obs(self, action, cost, readings, blocked, charged) -> Observation:
        return Observation(t=self.t, action=str(action), cost=round(cost, 6), battery=round(self.battery, 6),
                           pose=self.to_agent(self.pos), facing=self.facing, blocked=blocked,
                           on_hearth=self.on_hearth, charged=charged, readings=readings,
                           episode={"done": self.done, "reason": self.reason})

    # ---- truth for logs / viewer / evaluator (never given to the agent) --------------------------
    def truth_snapshot(self) -> dict:
        return {"t": self.t, "rescuer": list(self.pos), "facing": self.facing, "battery": self.battery,
                "target": list(self.target.pos), "energy": round(self.target.energy, 6), "alive": self.target.alive,
                "target_moved": self.target.moved_last,
                "remaining_life": round(self.target.remaining_life_estimate(), 2),
                "dogs": [[e.id, e.x, e.y] for e in self.entities if e.kind == "dog"]}

    def static_truth(self) -> dict:
        return {"spawn": list(self.spawn), "offset": list(self.offset), "life": self.life,
                "initial_signal": self.initial_signal, "signal_truth": self.signal_truth,
                "target_p0": self.target.p0, "target_d": self.target.d, "weather": self.weather,
                "entities": [{"id": e.id, "kind": e.kind, "pos": list(e.pos)} for e in self.entities],
                "rooms": self.map.rooms, "courtyards": self.map.courtyards}
