"""The rescuer agent. Sees only Observations; built only from the `agent` policy and `agent_assumed` sections.

Each turn: predict (target may move) -> ingest the observation (occupancy, pruning, Bayes update, tracks,
motion estimate) -> decide. Decision priority:
  charge (one turn, to 100%) when on a hearth and needed > return home when the safety rule says so >
  rescue (fresh CV `target` label) > approach-and-confirm a candidate > planner (best scan position x
  sensor) > recharge / frontier when nothing is affordable; then the safety rule / endgame override.
Watchdogs: approach progress (escalate routing, viewpoint, then give the candidate a cooldown) and
stagnation (no entropy drop, no new cells, no CV for a while). Every escalation is logged.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..common.actions import (CHARGE, CV, DIRS, LIDAR, MOVE, RADAR, THERMAL, TURN, WAIT, Action, action_cost,
                              rotate)
from ..common.geometry import disk_cells, line_of_sight
from ..common.observation import Observation
from . import charging, override, report, safety
from .belief import Belief, possible_mask
from .fields import bfs_field, cost_field, direction, path_down, path_to, step_down
from .frame import Frame
from .fusion import Tracker
from .motion_estimate import LifeModel, MotionEstimator, life_prior_mean
from .occupancy import FREE, UNKNOWN, WALL, Occupancy, outer_wall_mask
from .planner import Planner
from .sensors_assumed import SensorModels

RESCUE_WINDOW = 10   # a-priori rule: rescue counts if a CV `target` label is at most this many turns old
OVERRIDE_EVERY = 8   # re-evaluate an ongoing override / return decision this often
HOME_REFRESH = 25    # recompute the home field this often so the bound h never drifts far above the truth
SHORT_EXCURSION = 20


def cv_pattern(pose, facing: str) -> list[tuple[int, int]]:
    """Cells the camera looks at: the one in front, then the row of three beyond it."""
    dx, dy = DIRS[facing]
    lx, ly = -dy, dx
    return [(pose[0] + dx, pose[1] + dy)] + [(pose[0] + 2 * dx + k * lx, pose[1] + 2 * dy + k * ly) for k in (-1, 0, 1)]


class Agent:
    def __init__(self, policy: dict, assumed: dict, seed: int = 0):
        self.pol = policy
        self.a = assumed
        self.costs = assumed["costs"]
        self.frame = Frame(assumed["map_size"])
        n = self.frame.size
        self.occ = Occupancy(n)
        self.belief = Belief(n)
        self.models = SensorModels(assumed)
        self.tracker = Tracker()
        self.planner = Planner(self.models, self.costs, policy["planner"])
        self.rng = np.random.default_rng(seed)
        self.reserve = float(policy["reserve"])
        self.per_cell = float(policy["return_cost_per_cell"])
        self.temper = float(policy.get("evidence_temper", 1.0))
        self.ap = policy.get("approach", {})
        self.t = 0
        self.pose = tuple(self.frame.spawn)
        self.facing = "N"
        self.battery = float(self.costs["battery_start"])
        self.on_hearth = False
        self.h = 0
        self.home = np.full((n, n), -1, dtype=np.int32)
        self.home_t = -1
        self.field = None          # exploration field from field_origin
        self.field_origin = None
        self.route = None          # {"goal", "path", "origin", "purpose", "t", "conservative"}
        self.plan = None           # planner choice {"goal", "sensor", ...}
        self.plan_peak = None
        self.mode = "search"       # search | approach | return | dash | lean
        self.mode_t = 0
        self.scans = {"thermal": [], "radar": [], "lidar": [], "cv": []}
        self.last_cv_target = None  # (t, cell)
        self.overrides: list[dict] = []
        self.spend = defaultdict(float)
        self.recharges = 0
        self.outcome = None
        self.life = None
        self.motion = None
        self.decision: dict = {}
        self.fused: list[dict] = []
        self.decision_note = ""
        self.outside = np.zeros((n, n), dtype=bool)  # routing-only: probably beyond the outer wall
        self.approach = None        # current candidate: {"cell", "t0", "best", "best_t", "level", "bad"}
        self.cooldown: list[tuple[tuple[int, int], int]] = []
        self.wall_views: set = set()  # (pose, facing) whose CV saw only a wall
        self.excursions: list[list] = []  # [leave_t, back_t]
        self.escalations: list[dict] = []
        self.loop_events: list[dict] = []
        self.events: dict = {}
        self.history: list[tuple] = []  # (t, entropy, known cells, cv fired) for the stagnation watchdog
        self.safety_log: dict | None = None

    # =============================================================================================
    # observation ingestion
    # =============================================================================================
    def act(self, obs: Observation) -> Action | None:
        self.observe(obs)
        if obs.done:
            return None
        return self.decide()

    def possible(self) -> np.ndarray:
        return possible_mask(self.occ.grid)

    def _init(self, obs: Observation) -> None:
        self.models.set_weather(obs.weather)
        self.planner.reset_kernels()
        self.pose = tuple(obs.pose)
        self.facing = obs.facing
        self.battery = obs.battery
        self.on_hearth = obs.on_hearth
        self.occ.mark([self.pose], FREE)
        if obs.on_hearth:
            self.occ.hearths.add(self.pose)
        sig = obs.initial_signal
        sm = self.a["signal"]
        s = min(max(sig["strength"], 1e-4), 0.9999)
        d_est = sm["d0"] * math.sqrt(1 / s - 1)
        tgt = self.a["target"]
        self.life = LifeModel(life_prior_mean(d_est, tgt["k"]), self.a.get("life_prior_cv", 0.35))
        self.motion = MotionEstimator(tgt["p0"], self.life)
        self.belief.init_from_signal(self.pose, sig, sm, tgt["p0"], self.possible())
        self.signal = dict(sig, distance_est=round(d_est, 1))
        self._recompute_home()

    def observe(self, obs: Observation) -> None:
        self.t = obs.t
        self.fused = []
        self.events = {}
        if obs.t == 0:
            self._init(obs)
            return
        self.belief.predict(self.motion.estimate(self.t - 1), self.occ.walkable)
        act = Action.parse(obs.action)
        self._walls_changed = False
        moved = False
        new_hearth = False
        was_on_hearth = self.on_hearth
        if act.kind == MOVE:
            self.facing = act.arg
            dx, dy = DIRS[act.arg]
            nxt = (self.pose[0] + dx, self.pose[1] + dy)
            if obs.blocked:
                self._walls_changed |= self.occ.mark([nxt], WALL) > 0
            else:
                self.pose, moved = nxt, True
            # book the heading change under `turn`, the step itself under `move`
            self.spend["move"] += min(obs.cost, self.costs["move"])
            self.spend["turn"] += max(0.0, obs.cost - self.costs["move"])
        else:
            self.spend[act.kind] += obs.cost
        if act.kind == TURN:
            self.facing = rotate(self.facing, act.arg)
        assert tuple(obs.pose) == self.pose, "dead reckoning disagrees with reported pose"
        self.battery = obs.battery
        self.on_hearth = obs.on_hearth
        self.occ.mark([self.pose], FREE)
        inside = [self.pose]
        if obs.on_hearth and self.pose not in self.occ.hearths:
            self.occ.hearths.add(self.pose)
            new_hearth = True
        self._track_excursion(was_on_hearth, obs.on_hearth)
        known = self.tracker.known_decoys(self.t) + self.tracker.sources(self.t) + \
            [{"kind": "hearth", "pos": list(h), "sigma": 0.3} for h in self.occ.hearths]
        feas = self.frame.mask()
        r = obs.readings
        if "lidar" in r:
            L = r["lidar"]
            self.occ.mark(L["free"], FREE)
            self._walls_changed |= self.occ.mark(L["walls"], WALL) > 0
            inside += [tuple(c) for c in L["free"]] + [tuple(c) for c in L["walls"]]
            self.belief.apply(*self.models.lidar_likelihood(L["free"], L["blobs"], feas, known), self.temper)
            for b in L["blobs"]:
                tr = self.tracker.add(self.t, "lidar", b["pos"], 0.3, attr=b["shape"])
                self.fused.append(tr.to_dict())
                if b["shape"] == "box" and tuple(b["pos"]) not in self.occ.hearths:
                    self.occ.hearths.add(tuple(b["pos"]))
                    new_hearth = True
            self.scans["lidar"].append((self.t, self.pose))
        if "radar" in r:
            self.belief.apply(*self.models.radar_likelihood(self.pose, r["radar"], self.occ, feas, known), self.temper)
            for b in r["radar"]:
                self.fused.append(self.tracker.add(self.t, "radar", b["pos"], b["radius"] / math.sqrt(3)).to_dict())
            self.scans["radar"].append((self.t, self.pose))
        if "thermal" in r:
            self.belief.apply(*self.models.thermal_likelihood(self.pose, r["thermal"], self.occ, feas, known), self.temper)
            for b in r["thermal"]:
                self.fused.append(self.tracker.add(self.t, "thermal", b["pos"], max(1.0, b["radius"] * 0.8),
                                                   strength=b["strength"]).to_dict())
            self.scans["thermal"].append((self.t, self.pose))
        if "cv" in r:
            C = r["cv"]
            self.occ.mark(C["visible"], FREE)
            self._walls_changed |= self.occ.mark(C["walls"], WALL) > 0
            inside += [tuple(c) for c in C["visible"]] + [tuple(c) for c in C["walls"]]
            self.belief.apply(*self.models.cv_likelihood(C["visible"], C["labels"], feas, known), self.temper)
            for lab in C["labels"]:
                self.fused.append(self.tracker.add(self.t, "cv", lab["pos"], 0.3, label=lab["label"]).to_dict())
                if lab["label"] == "target":
                    self.last_cv_target = (self.t, tuple(lab["pos"]))
                if lab["label"] == "hearth" and tuple(lab["pos"]) not in self.occ.hearths:
                    self.occ.hearths.add(tuple(lab["pos"]))
                    new_hearth = True
            if not C["visible"]:
                self.wall_views.add((self.pose, self.facing))
            self.scans["cv"].append((self.t, self.pose, self.facing))
            self.events["cv"] = {"pattern": [list(c) for c in cv_pattern(self.pose, self.facing)],
                                 "visible": C["visible"], "walls": C["walls"], "labels": C["labels"]}
        if self.frame.note_inside(inside):
            self.occ.apply_window(self.frame.mask())
            self._walls_changed = True
        if self._walls_changed:
            self.field = None  # the exploration field must not route through newly known walls
            if self.pol.get("outer_wall", True):
                self.outside = outer_wall_mask(self.occ)
        # entering a cell with a fresh CV label and not being rescued: she is not in *this* cell, but she
        # may have just stepped, so its mass goes to the walkable neighbours instead of vanishing
        if moved and self.last_cv_target and self.t - self.last_cv_target[0] <= RESCUE_WINDOW:
            self._failed_entry()
        self.belief.normalise(self.possible())
        if moved:
            hv = self.home[self.pose]
            self.h = min(self.h + 1, int(hv)) if hv >= 0 else self.h + 1
        if self.pose in self.occ.hearths:
            self.h = 0
        if new_hearth or self.t - self.home_t >= HOME_REFRESH:
            self._recompute_home()
        for tr in self.tracker.recent(self.t, 0):
            if self._target_like(tr):
                self.motion.add_track(tr, self.t)
        self.history.append((self.t, self.belief.entropy_bits(), int((self.occ.grid != UNKNOWN).sum()), act.kind == CV))
        self.history = self.history[-60:]
        if obs.done:
            self.outcome = obs.episode["reason"]

    def _failed_entry(self) -> None:
        x, y = self.pose
        m = float(self.belief.p[x, y])
        self.belief.p[x, y] = 0.0
        nbs = [(x + dx, y + dy) for dx, dy in DIRS.values() if self.occ.walkable[x + dx, y + dy]]
        for c in nbs:
            self.belief.p[c] += m / len(nbs)
        self.events["failed_entry"] = {"cell": [x, y], "moved_mass": round(m, 4)}

    def _track_excursion(self, was: bool, now: bool) -> None:
        if was and not now:
            self.excursions.append([self.t, None])
        elif now and not was and self.excursions and self.excursions[-1][1] is None:
            self.excursions[-1][1] = self.t
            done = [e for e in self.excursions if e[1] is not None]
            if len(done) >= 2 and all(e[1] - e[0] < SHORT_EXCURSION for e in done[-2:]):
                ev = {"t": self.t, "excursions": [list(e) for e in done[-2:]], "battery": self.battery, "h": self.h}
                self.loop_events.append(ev)
                self.events["loop_guard"] = ev

    def _target_like(self, tr) -> bool:
        """Only tracks that are the target by CV, or carry both a heartbeat and a humanoid shape, may
        inform the movement estimate (a static suit of armour must not read as an exhausted target)."""
        if tr.kind == "target":
            return True
        return tr.kind == "unknown" and tr.hits.get("humanoid", 0) > 0 and tr.hits.get("radar", 0) > 0

    # =============================================================================================
    # fields and routes
    # =============================================================================================
    def _window_field(self, passable: np.ndarray, sources: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = self.frame.window
        out = np.full(passable.shape, -1, dtype=np.int32)
        out[x0:x1 + 1, y0:y1 + 1] = bfs_field(passable[x0:x1 + 1, y0:y1 + 1], sources[x0:x1 + 1, y0:y1 + 1])
        return out

    def _conservative_field(self, sources: np.ndarray) -> np.ndarray:
        """Cost field where unknown cells are expensive, so routes prefer floor (and doors) already seen."""
        x0, x1, y0, y1 = self.frame.window
        g = self.occ.grid[x0:x1 + 1, y0:y1 + 1]
        cost = np.where(g == FREE, 1.0, np.where(g == UNKNOWN, float(self.ap.get("unknown_cost", 4.0)), np.inf))
        cost[self.outside[x0:x1 + 1, y0:y1 + 1]] = np.inf
        out = np.full(self.occ.grid.shape, np.inf)
        out[x0:x1 + 1, y0:y1 + 1] = cost_field(cost, sources[x0:x1 + 1, y0:y1 + 1])
        return out

    def _recompute_home(self) -> None:
        self.home = self._window_field(self.occ.free, self.occ.hearth_mask())
        self.home_t = self.t
        hv = self.home[self.pose]
        if hv >= 0:
            self.h = min(self.h, int(hv)) if self.t > 0 else int(hv)

    def _explore_field(self) -> np.ndarray:
        """Route cost from the agent's cell: BFS with unknown cells treated as free. Unreachable: -1."""
        if self.field is None or self.field_origin != self.pose:
            src = np.zeros(self.occ.grid.shape, dtype=bool)
            src[self.pose] = True
            self.field = self._window_field(self.occ.walkable & ~self.outside, src)
            self.field_origin = self.pose
        return self.field

    @staticmethod
    def _reachable(field, c) -> bool:
        v = field[c]
        return bool(np.isfinite(v) and v >= 0)

    def _path(self, goal):
        f = self._explore_field()
        return path_to(f, goal)

    def nearest_hearth(self):
        if not self.occ.hearths:
            return None
        return min(self.occ.hearths, key=lambda c: abs(c[0] - self.pose[0]) + abs(c[1] - self.pose[1]))

    def _goto(self, goal, purpose: str, conservative: bool = False) -> Action | None:
        """Next move towards goal; replans on a new goal, a wall on the path or a routing-mode change."""
        goal = tuple(goal)
        if goal == self.pose:
            return None
        r = self.route
        need = r is None or r["goal"] != goal or r.get("conservative") != conservative or \
            (self.pose not in r["path"] and self.pose != r["origin"])
        if not need:
            i = r["path"].index(self.pose) + 1 if self.pose in r["path"] else 0
            nxt = r["path"][i] if i < len(r["path"]) else None
            need = nxt is None or any(self.occ.grid[c] == WALL for c in r["path"][i:])
        if need:
            if conservative:
                src = np.zeros(self.occ.grid.shape, dtype=bool)
                src[self.pose] = True
                path = path_down(self._conservative_field(src), goal)
            else:
                path = self._path(goal)
            if not path:
                self.route = None
                return None
            if r is not None and r["goal"] != goal:
                self.events["reset"] = {"from": list(r["goal"]), "to": list(goal), "trigger": f"new goal ({purpose})"}
            self.route = r = {"goal": goal, "path": path, "origin": self.pose, "purpose": purpose, "t": self.t,
                              "conservative": conservative}
            i = 0
        return Action(MOVE, direction(self.pose, r["path"][i]))

    def _remaining_route(self) -> int:
        r = self.route
        if r is None:
            return 10 ** 6
        return len(r["path"]) - (r["path"].index(self.pose) + 1 if self.pose in r["path"] else 0)

    def _face(self, want: str) -> Action | None:
        if self.facing == want:
            return None
        return Action(TURN, "R" if rotate(self.facing, "R") == want else "L")

    # =============================================================================================
    # decision
    # =============================================================================================
    def decide(self) -> Action:
        self.safety_log = None
        action, reason, extra = self._policy()
        cost = action_cost(action, self.facing, self.costs)
        if cost > self.battery + 1e-9:
            action, reason = Action(WAIT), "battery too low for any action; waiting (logged)"
        bs = self.belief.summary()
        self.decision = {
            "action": str(action), "reason": reason, "mode": self.mode,
            "plan": self._plan_summary(), "override": extra.get("override"), "h": int(self.h),
            "safety": self.safety_log, "candidates": (self.plan or {}).get("candidates"),
            "approach": self._approach_summary(), "events": self.events or None,
            "escalation": self.events.get("escalation"),
            "belief": bs, "p_move_estimate": round(self.motion.estimate(self.t), 4),
            "energy_estimate": round(self.motion.energy_estimate(self.t), 3),
            "remaining_life": self.life.remaining(self.t), "fused": self.fused,
            "window": list(self.frame.window),
            "hearths": sorted(list(h) for h in self.occ.hearths),
        }
        return action

    def _plan_summary(self) -> dict | None:
        if self.route is None and self.plan is None:
            return None
        out = {}
        if self.plan:
            out.update({k: self.plan[k] for k in ("goal", "sensor", "score", "q") if k in self.plan})
        if self.route:
            out["route_goal"] = list(self.route["goal"])
            out["purpose"] = self.route["purpose"]
            out["conservative"] = self.route.get("conservative", False)
            i = self.route["path"].index(self.pose) + 1 if self.pose in self.route["path"] else 0
            out["path"] = [list(c) for c in self.route["path"][i:i + 60]]
        return out

    def _approach_summary(self) -> dict | None:
        a = self.approach
        if a is None:
            return None
        return {"cell": list(a["cell"]), "since": a["t0"], "level": a["level"], "best_remaining": a["best"],
                "view": a.get("view")}

    def _escalate(self, kind: str, detail: str) -> None:
        ev = {"t": self.t, "kind": kind, "detail": detail}
        self.escalations.append(ev)
        self.events["escalation"] = ev

    def _set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.mode_t = self.t
            self.route = None
            self.plan = None

    def _policy(self) -> tuple[Action, str, dict]:
        extra: dict = {}
        # ---- charging: one turn to 100%, only when needed or already standing on a hearth --------------
        if self.on_hearth:
            required = self.mode == "return"
            do_charge = charging.should_charge(self.battery, True, required,
                                               self.pol["charging"]["opportunistic_below"])
            if do_charge:
                why = "the safety rule sent us here" if required else (
                    f"standing on a hearth at {self.battery:.1f}% (< {self.pol['charging']['opportunistic_below']:.0f}%)")
                self.recharges += 1
                self._set_mode("search")
                return Action(CHARGE), f"charge to 100% in one turn: {why}", extra
            if self.mode == "return":
                self._set_mode("search")
        if self.mode == "return":
            if self.t - self.mode_t >= OVERRIDE_EVERY and self.pol["override"]["enabled"]:
                ov = self._run_override()
                extra["override"] = ov
                if ov["choice"] != "safe":
                    self._set_mode(ov["choice"])
            if self.mode == "return":
                return self._step_home("returning to recharge (safety rule)"), self.decision_note, extra
        if self.mode in ("dash", "lean") and self.t - self.mode_t >= OVERRIDE_EVERY:
            ov = self._run_override(current=self.mode)
            extra["override"] = ov
            if ov["choice"] == "safe":
                self._set_mode("return")
                return self._step_home("override re-check: safe plan is better again"), self.decision_note, extra
        action, reason = self._search_action()
        if action.kind == CHARGE or (self.mode == "return" and action.kind == MOVE):
            return action, reason, extra
        # ---- safety rule -------------------------------------------------------------------
        if self.mode in ("dash", "lean"):
            return action, f"[{self.mode} override] " + reason, extra
        h_after = self._h_after(action)
        cost = action_cost(action, self.facing, self.costs)
        if safety.near_failing(self.battery, cost, h_after, self.reserve, self.per_cell) and self.home_t < self.t:
            self._recompute_home()
            h_after = self._h_after(action)
        # the last confirm steps (face, CV, step into her cell) may spend the reserve: the way home is
        # known floor and 0.4%/cell already covers the worst-case turn, so only the slack is at stake
        finishing = (self.mode == "approach" and action.kind in (CV, TURN) and self.approach is not None
                     and self.approach.get("view_cell") is not None and self.approach["view_cell"][0] == self.pose) \
            or (self.route is not None and self.route["purpose"] == "rescue" and action.kind == MOVE)
        reserve = 0.0 if finishing else self.reserve
        margin = safety.margin(self.battery, cost, h_after, reserve, self.per_cell)
        self.safety_log = {"battery": round(self.battery, 2), "cost": round(cost, 2), "h_after": int(h_after),
                           "reserve": reserve, "margin": round(margin, 2), "finishing": finishing,
                           "near_failing": bool(safety.near_failing(self.battery, cost, h_after, self.reserve, self.per_cell))}
        if margin < -1e-9:
            if self.pol["override"]["enabled"]:
                ov = self._run_override()
                extra["override"] = ov
                if ov["choice"] != "safe":
                    self._set_mode(ov["choice"])
                    action, reason = self._search_action()
                    return action, f"[{self.mode} override] " + reason, extra
            self._set_mode("return")
            if self.on_hearth:
                self.recharges += 1
                self._set_mode("search")
                return Action(CHARGE), "charge to 100% in one turn: the safety rule needs it", extra
            return self._step_home(f"safety rule: {action} would leave too little to get home "
                                   f"({self.battery:.1f}% vs {self.per_cell}%x{h_after}+{self.reserve}%)"), \
                self.decision_note, extra
        return action, reason, extra

    def _h_after(self, action: Action) -> int:
        if action.kind != MOVE:
            return self.h
        dx, dy = DIRS[action.arg]
        nxt = (self.pose[0] + dx, self.pose[1] + dy)
        hv = self.home[nxt]
        return self.h - 1 if 0 <= hv < self.h else self.h + 1

    def _step_home(self, why: str) -> Action:
        if self.home[self.pose] < 0 or self.home[self.pose] > self.h:
            self._recompute_home()
        nxt = step_down(self.home, self.pose)
        if nxt is None:
            self._recompute_home()
            nxt = step_down(self.home, self.pose)
        self.decision_note = f"{why}; {self.h} moves to the nearest known hearth"
        if nxt is None:
            if self.on_hearth:
                return Action(CHARGE)
            self.decision_note += " (no known route home: waiting)"
            return Action(WAIT)
        return Action(MOVE, direction(self.pose, nxt))

    # ---- search behaviour -------------------------------------------------------------------
    def _search_action(self) -> tuple[Action, str]:
        self._stagnation_watchdog()
        action, reason = self._search_core()
        # never walk blind into unmapped space: LIDAR (0.2%) first when the surroundings are unknown
        if action.kind == MOVE and not (self.route and self.route["purpose"] == "rescue") and self.mode != "return":
            p = self.pol["planner"]
            unknown_near = self._unknown_near()
            if unknown_near >= p["lidar_unknown_trigger"] and \
                    not self._recent_scan("lidar", self.pose, p["lidar_spacing"], 10 ** 6):
                return Action(LIDAR), f"{unknown_near} unmapped cells within LIDAR range: map before moving on"
            dx, dy = DIRS[action.arg]
            if self.occ.grid[self.pose[0] + dx, self.pose[1] + dy] == UNKNOWN and \
                    not self._recent_scan("lidar", self.pose, 0, 10 ** 6):
                return Action(LIDAR), "next step is unmapped: LIDAR rather than bump into a wall"
        return action, reason

    def _stagnation_watchdog(self) -> None:
        """No entropy drop, no newly known cells and no CV for `stagnation_turns`: force a change."""
        n = int(self.pol.get("stagnation_turns", 15))
        if n <= 0:
            return
        hist = [h for h in self.history if self.t - h[0] <= n]
        if len(hist) < n or self.mode not in ("search", "approach"):
            return
        if self.escalations and self.t - self.escalations[-1]["t"] < n:
            return
        e_drop = hist[0][1] - hist[-1][1]
        new_cells = hist[-1][2] - hist[0][2]
        if e_drop < 0.05 and new_cells == 0 and not any(h[3] for h in hist):
            top, mass = self.belief.peaks(1)[0]
            if mass >= 0.1 and not self._cooling(top):
                self.approach = self._new_approach(top)
                self.approach["level"] = 1
                self._escalate("stagnation", f"no progress for {n} turns: forced approach to {list(top)} "
                                             f"(mass {mass:.2f}) with conservative routing")
            else:
                self.plan = None
                self._escalate("stagnation", f"no progress for {n} turns: replan, frontier if nothing scores")

    def _cooling(self, cell) -> bool:
        self.cooldown = [(c, u) for c, u in self.cooldown if u > self.t]
        return any(math.dist(c, cell) <= 3 for c, u in self.cooldown)

    def _new_approach(self, cell) -> dict:
        return {"cell": tuple(cell), "t0": self.t, "best": 10 ** 6, "best_t": self.t, "level": 0, "bad": set()}

    def _pick_candidate(self):
        """Keep pursuing the current candidate (re-aimed at where the belief says she is now) unless it
        faded; otherwise start on the strongest new one."""
        thr = self.pol["planner"]["confirm_mass"]
        a = self.approach
        if a is not None:
            x, y = a["cell"]
            r = int(self.ap.get("view_radius", 4))
            sub = self.belief.p[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1]
            j = np.unravel_index(int(np.argmax(sub)), sub.shape)
            new = (max(0, x - r) + int(j[0]), max(0, y - r) + int(j[1]))
            mass = self.belief.mass_near(new, 2)
            fresh = any(math.dist(tr.pos, new) <= 3 for tr in self._signature_tracks())
            if mass >= 0.04 or fresh or self.mode in ("dash", "lean"):
                if new != a["cell"]:
                    a["cell"] = new  # pursue: re-aim, keep the sequence and its progress
                return new, mass
            self.approach = None
        peaks = self.belief.peaks(3)
        for cell, m in peaks:
            if (m >= thr or self.mode in ("dash", "lean")) and not self._cooling(cell):
                self.approach = self._new_approach(cell)
                return cell, m
        for tr in self._signature_tracks():
            cell = (int(round(tr.pos[0])), int(round(tr.pos[1])))
            if not self._cooling(cell):
                self.approach = self._new_approach(cell)
                return cell, self.belief.mass_near(cell, 2)
        return None

    def _search_core(self) -> tuple[Action, str]:
        # rescue: a fresh CV label says the target is right there
        if self.last_cv_target and self.t - self.last_cv_target[0] < RESCUE_WINDOW:
            cell = self.last_cv_target[1]
            if self.belief.mass_near(cell, 1) > 0.05 and cell != self.pose:
                a = self._goto(cell, "rescue")
                if a:
                    return a, f"CV labelled the target at {list(cell)} {self.t - self.last_cv_target[0]} turns ago: moving in to rescue"
        cand = self._pick_candidate()
        if cand is not None:
            cell, m = cand
            if self.mode == "dash" and not self._recent_scan("thermal", cell, self.pol["planner"]["thermal_repeat_radius"], 40) \
                    and math.dist(self.pose, cell) <= 4:
                return Action(THERMAL), f"dash: thermal on arrival near {list(cell)}"
            a = self._confirm(cell)
            if a is not None:
                if self.mode == "search":
                    self.mode = "approach"
                return a, f"confirm: candidate at {list(cell)} holds {m:.2f} of the belief: {self.decision_note}"
        if self.mode == "approach":
            self.mode = "search"
        # planner
        self._maybe_replan()
        if self.plan is None:
            return self._nothing_affordable()
        goal = self.plan["goal"]
        if goal == self.pose:
            s = self.plan["sensor"]
            plan = self.plan
            self.plan = None
            return Action(s), (f"{s.upper()} here: expected detection mass {plan['q']:.2f}, "
                               f"{plan['gain']:.2f} bits for {self.costs[s]}%")
        a = self._goto(goal, f"{self.plan['sensor']} scan")
        if a is None:
            self.events["reset"] = {"from": list(goal), "to": None, "trigger": "planned position unreachable"}
            self.plan = None
            self._maybe_replan()
            if self.plan is None or self.plan["goal"] == self.pose:
                return self._nothing_affordable()
            a = self._goto(self.plan["goal"], f"{self.plan['sensor']} scan")
            if a is None:
                return self._nothing_affordable()
        return a, (f"moving to {list(self.plan['goal'])} ({self.plan['dist']} moves) for a {self.plan['sensor']} scan "
                   f"(score {self.plan['score']:.3f}, q={self.plan['q']:.2f})")

    def _nothing_affordable(self) -> tuple[Action, str]:
        """No scan position scores (usually: not enough battery for any trip). Never idle: recharge if
        that helps, otherwise push the map frontier, otherwise wait with the reason logged."""
        if self.mode in ("dash", "lean"):
            top = self.belief.peaks(1)[0][0]
            a = self._goto(top, "dash")
            if a:
                return a, f"heading for the most likely cell {list(top)}"
        if self.battery < 99.0 and self.occ.hearths and self.mode == "search":
            self._set_mode("return")
            if self.on_hearth:
                self.recharges += 1
                self._set_mode("search")
                return Action(CHARGE), "no affordable scan: charge to 100% here"
            return self._step_home("no affordable scan from here: recharge first"), self.decision_note
        a = self._frontier()
        if a is not None:
            return a, "nothing scores: push the nearest map frontier"
        return Action(WAIT), "nothing affordable, nothing unexplored in reach: waiting (logged)"

    def _frontier(self) -> Action | None:
        g = self.occ.grid
        unk = g == UNKNOWN
        nb = np.zeros_like(unk)
        nb[1:, :] |= unk[:-1, :]
        nb[:-1, :] |= unk[1:, :]
        nb[:, 1:] |= unk[:, :-1]
        nb[:, :-1] |= unk[:, 1:]
        front = (g == FREE) & nb & ~self.outside & self.frame.mask()
        field = self._explore_field()
        ok = front & np.isfinite(field) & (field > 0)
        if not ok.any():
            return None
        idx = np.argwhere(ok)
        j = int(np.argmin(field[idx[:, 0], idx[:, 1]]))
        return self._goto(tuple(int(v) for v in idx[j]), "frontier")

    def _signature_tracks(self) -> list:
        """Unidentified sources worth a CV look: humanoid + warm/heartbeat with some belief on them, or any
        fresh heartbeat close by (only the target and a dog or two have one, so checking is cheap)."""
        out = []
        for tr in self.tracker.recent(self.t, 10):
            if tr.kind != "unknown":
                continue
            h = tr.hits
            cell = (round(tr.pos[0]), round(tr.pos[1]))
            signature = h.get("humanoid") and (h.get("radar") or h.get("thermal")) and self.belief.mass_near(cell, 2) > 0.02
            heartbeat_near = h.get("radar") and math.dist(self.pose, tr.pos) <= 12
            if signature or heartbeat_near:
                out.append(tr)
        return sorted(out, key=lambda tr: -self.belief.mass_near((round(tr.pos[0]), round(tr.pos[1])), 2))

    def _recent_scan(self, sensor: str, where, radius: float, age: int) -> bool:
        return any(self.t - s[0] <= age and math.dist(s[1], where) <= radius for s in self.scans[sensor])

    def _unknown_near(self) -> int:
        cells = disk_cells(self.pose[0], self.pose[1], self.models.lidar_range(), self.occ.grid.shape)
        return int((self.occ.grid[cells[:, 0], cells[:, 1]] == UNKNOWN).sum())

    # ---- approach and confirm ---------------------------------------------------------------
    def _view_mass(self, stand, facing) -> float:
        """Belief mass the CV pattern would cover from `stand` facing `facing`, with line of sight in the
        agent's own map (unknown cells count as clear, known walls block)."""
        if (stand, facing) in self.wall_views:
            return 0.0
        pat = cv_pattern(stand, facing)
        n = self.occ.grid.shape[0]
        front = pat[0]
        if not (0 <= front[0] < n and 0 <= front[1] < n) or self.occ.grid[front] == WALL:
            return 0.0
        m = float(self.belief.p[front])
        far = [c for c in pat[1:] if 0 <= c[0] < n and 0 <= c[1] < n]
        if far:
            for c, ok in zip(far, line_of_sight(self.occ.wall, stand, far)):
                if ok and self.occ.grid[c] != WALL:
                    m += float(self.belief.p[c])
        return m

    def _view_cost(self, travel, turns) -> float:
        """Cost of getting to a viewpoint and looking: turns, +2 for the CV turn and slack."""
        return travel + turns + 2

    def _choose_view(self, cell) -> tuple | None:
        """Standing cell + facing that cover the most belief per turn spent getting there."""
        a = self.approach
        r = int(self.ap.get("view_radius", 4))
        field = self._explore_field()
        best = None
        recent = {(s[1], s[2]) for s in self.scans["cv"] if self.t - s[0] <= 3}
        for x in range(cell[0] - r, cell[0] + r + 1):
            for y in range(cell[1] - r, cell[1] + r + 1):
                st = (x, y)
                if not (0 <= x < self.occ.grid.shape[0] and 0 <= y < self.occ.grid.shape[1]):
                    continue
                if st == cell or self.occ.grid[st] not in (FREE, UNKNOWN) or self.outside[st] or not self._reachable(field, st):
                    continue
                if a is not None and st in a["bad"]:
                    continue
                for f in DIRS:
                    if (st, f) in recent:
                        continue
                    m = self._view_mass(st, f)
                    if m <= 1e-6:
                        continue
                    turns = 0 if (st == self.pose and f == self.facing) else 1
                    score = m / self._view_cost(field[st], turns)
                    if best is None or score > best[0]:
                        best = (score, st, f, m)
        # hysteresis: stick with the current viewpoint unless it broke or a clearly better one appeared,
        # otherwise every newly seen wall re-ranks the views and the agent circles the room
        cur = a.get("view_cell") if a is not None else None
        if cur is not None and best is not None:
            st, f = cur
            ok = st != cell and self.occ.grid[st] in (FREE, UNKNOWN) and self._reachable(field, st) and \
                st not in a["bad"] and (st, f) not in recent and math.dist(st, cell) <= r + 1
            m = self._view_mass(st, f) if ok else 0.0
            if m > 1e-6:
                score = m / self._view_cost(field[st], 0 if (st == self.pose and f == self.facing) else 1)
                if best[0] < float(self.ap.get("view_hysteresis", 1.5)) * score:
                    best = (score, st, f, m)
        if best is not None and a is not None:
            a["view_cell"] = (best[1], best[2])
        return best

    def _confirm(self, cell) -> Action | None:
        """Approach-and-confirm: LIDAR, radar, then the best line-of-sight viewpoint, face, CV."""
        cell = tuple(cell)
        a = self.approach or self._new_approach(cell)
        self.approach = a
        if math.dist(self.pose, cell) <= self.models.lidar_range() * 0.9 and \
                not self._recent_scan("lidar", self.pose, 2.5, 8) and not self._recent_scan("lidar", cell, 4, 3):
            self.decision_note = "LIDAR first: exact shapes and the walls around it"
            return Action(LIDAR)
        if math.dist(self.pose, cell) <= self.models.s["radar"]["range"] * 0.6 and \
                not self._recent_scan("radar", cell, self.models.s["radar"]["range"] * 0.6, 20):
            self.decision_note = "radar first: does it have a heartbeat?"
            return Action(RADAR)
        view = self._choose_view(cell)
        if view is None:
            self._escalate("no_view", f"no reachable viewpoint with line of sight onto {list(cell)}: cooldown")
            self.cooldown.append((cell, self.t + int(self.ap.get("cooldown", 40))))
            self.approach = None
            return None
        _, stand, want, mass = view
        a["view"] = [list(stand), want, round(mass, 4)]
        if stand == self.pose:
            t = self._face(want)
            if t is not None:
                self.decision_note = f"turn to face {want} (pattern covers {mass:.2f} of the belief)"
                return t
            self.decision_note = f"CV to confirm (pattern covers {mass:.2f} of the belief)"
            return Action(CV)
        act = self._goto(stand, "confirm", conservative=a["level"] >= 1)
        rem = self._remaining_route()
        if rem < a["best"]:
            a["best"], a["best_t"] = rem, self.t
        elif self.ap.get("watchdog", True) and self.t - a["best_t"] >= int(self.ap.get("stall_turns", 12)):
            a["level"] += 1
            a["best_t"] = self.t
            a["best"] = rem
            if a["level"] == 1:
                self._escalate("approach_stall", f"no progress to {list(stand)} for {self.ap.get('stall_turns', 12)} "
                                                 "turns: route through known floor and doors (unknown cells cost more)")
                act = self._goto(stand, "confirm", conservative=True)
            elif a["level"] == 2:
                a["bad"].add(stand)
                self._escalate("approach_stall", f"still no progress: give up viewpoint {list(stand)}, pick another")
                return self._confirm(cell)
            else:
                self.cooldown.append((cell, self.t + int(self.ap.get("cooldown", 40))))
                self._escalate("approach_abandon", f"approach to {list(cell)} stalled 3 times: cooldown, search on")
                self.approach = None
                return None
        if act is None:
            a["bad"].add(stand)
            self.decision_note = f"viewpoint {list(stand)} unreachable"
            return None
        self.decision_note = f"approach {list(stand)} facing {want} ({rem} moves; pattern covers {mass:.2f})"
        return act

    def _best_plan(self, ignore_battery: bool = False) -> dict | None:
        field = self._explore_field()
        extra = [c for c, _ in self.belief.peaks(3)]
        return self.planner.choose(self.belief.p, self.frame.window, field, self.home, self.h, self.battery,
                                   self.pose, self.scans, self.t, self.reserve, self.per_cell, extra,
                                   ignore_battery=ignore_battery)

    def _maybe_replan(self) -> None:
        peak = tuple(self.belief.summary()["peak"])
        p = self.pol["planner"]
        trigger = "no plan"
        if self.plan is not None:
            age = self.t - self.plan["t"]
            stale = age > 30
            # hysteresis: a peak shift only counts once the plan has had a few turns to work
            shifted = age >= 5 and self.plan_peak is not None and math.dist(peak, self.plan_peak) > p["replan_peak_shift"]
            walled = self.occ.grid[self.plan["goal"]] == WALL
            if not (stale or shifted or walled):
                return
            trigger = "stale" if stale else "belief peak moved" if shifted else "goal became a wall"
        old = self.plan
        self.plan = self._best_plan(ignore_battery=self.mode in ("dash", "lean"))
        if self.plan is not None:
            self.plan["t"] = self.t
            if old is not None:
                self.events["reset"] = {"from": list(old["goal"]), "to": list(self.plan["goal"]), "trigger": trigger}
        self.plan_peak = peak

    # ---- override ---------------------------------------------------------------------------
    def _route_leg(self, field: np.ndarray, c) -> override.Leg:
        path = path_to(field, c) if field[c] >= 0 else None
        if path is None:
            return override.Leg(known=1e6, unknown=0)
        unk = sum(1 for p in path if self.occ.grid[p] == UNKNOWN)
        return override.Leg(known=len(path) - unk, unknown=unk)

    def _greedy_tour(self, start: np.ndarray, peaks) -> tuple[list, list, list]:
        """Visit order by mass per distance (nearest promising first), with the legs of that order."""
        walk = self.occ.walkable & ~self.outside
        fields = {i: self._window_field(walk, self._point(c)) for i, (c, _) in enumerate(peaks)}
        cur = self._window_field(walk, start)
        left = list(range(len(peaks)))
        order, legs = [], []
        while left:
            def value(i):
                d = cur[peaks[i][0]]
                return -1 if d < 0 else peaks[i][1] / (1 + d / 15.0)
            i = max(left, key=value)
            legs.append(self._route_leg(cur, peaks[i][0]))
            order.append(i)
            left.remove(i)
            cur = fields[i]
        return [peaks[i][0] for i in order], [peaks[i][1] for i in order], legs

    def _point(self, c) -> np.ndarray:
        m = np.zeros(self.occ.grid.shape, dtype=bool)
        m[tuple(c)] = True
        return m

    def _run_override(self, current: str | None = None) -> dict:
        po = self.pol["override"]
        peaks = self.belief.peaks(int(po["locations"]))
        locs, masses, dash_legs = self._greedy_tour(self._point(self.pose), peaks)
        slocs, smasses, safe_legs = self._greedy_tour(self.occ.hearth_mask(), peaks)
        params = dict(po, hysteresis=self.pol["hysteresis"])
        p_confirm = float(po.get("p_confirm", 0.85))
        ov = override.evaluate(self.t, self.battery, masses, dash_legs, float(self.h), safe_legs, self.costs,
                               self.life, self.rng, params, self.reserve, self.per_cell,
                               p_move=self.motion.estimate(self.t), p_confirm=p_confirm)
        # the safe tour has its own visiting order; re-score it with its own masses
        sv = override.evaluate(self.t, self.battery, smasses, dash_legs, float(self.h), safe_legs, self.costs,
                               self.life, self.rng, params, self.reserve, self.per_cell,
                               p_move=self.motion.estimate(self.t), p_confirm=p_confirm)
        ov["plans"]["safe"] = sv["plans"]["safe"]
        ov["detail"]["safe"] = sv["detail"]["safe"]
        best_risky = max(ov["plans"]["dash"], ov["plans"]["lean"])
        risky = "dash" if ov["plans"]["dash"] >= ov["plans"]["lean"] + 1e-9 else "lean"
        if current in ("dash", "lean"):
            # hysteresis in favour of the plan already being executed
            ov["choice"] = "safe" if ov["plans"]["safe"] >= best_risky + self.pol["hysteresis"] else current
        else:
            ov["choice"] = risky if best_risky >= ov["plans"]["safe"] + self.pol["hysteresis"] else "safe"
        ov["p_chosen"] = ov["plans"][ov["choice"]]
        ov["locations"] = [[list(c), round(m, 4)] for c, m in zip(locs, masses)]
        self.overrides.append({k: ov[k] for k in ("t", "plans", "choice", "battery_left", "p_chosen")})
        self.mode_t = self.t
        return ov

    # ---- report -----------------------------------------------------------------------------
    def final_report(self) -> dict:
        return report.build(self)
