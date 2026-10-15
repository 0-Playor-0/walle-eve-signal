"""Endgame risk override: rescue is the primary goal, survival secondary.

Runs only when the safety rule would force a recharge. Three plans are scored:
    P(rescue | plan) = sum_l [belief(l) x retained(dt_l)]^w_loc x P(alive at arrival)^w_alive
                             x P(battery lasts to l)^w_batt x P(confirm on arrival)
  safe : walk home, charge (one turn, to 100%), then visit the locations (thermal + CV each)
  dash : go now, thermal + CV at each location, no return reserve
  lean : go now, CV only at each location
- The battery term: the known part of each leg is exact; the unknown part is sampled as a detour
  (length x (1 + Exp(mean 0.5))) `samples` times, and the affordable fraction is used.
- retained(dt): the target keeps moving while the agent is away, so a location's mass decays with the
  delay: the 2-cell box (25 cells) against a random walk of variance p_move x dt,
  retained = 25 / (25 + 2 pi p_move dt). Without it, Safe looked as good as finishing now (seed 1).
- P(confirm on arrival): reaching a location is not yet a rescue (CV miss, target one cell off the pattern).
- Safe is worth at least restart_value x P(alive when back out): with a full battery the agent can search
  again, which a dash that strands it gives up.
Locations are visited greedily by mass per distance, not by mass alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Leg:
    known: float     # cells through known-free territory (exact)
    unknown: float   # cells through unknown territory (optimistic; sampled as a detour)


def retained(dt, p_move: float):
    dt = np.maximum(np.asarray(dt, dtype=np.float64), 0.0)
    return 25.0 / (25.0 + 2 * math.pi * max(p_move, 0.0) * dt)


def _tour(battery0: float, t0: float, legs: list[Leg], masses: list[float], scan_cost: float, scan_turns: int,
          per_cell: float, life, rng: np.random.Generator, samples: int, w: tuple, p_move: float, t_now: float,
          p_confirm: float):
    wl, wa, wb = w
    cost = np.zeros(samples)
    turns = np.zeros(samples)
    total = 0.0
    detail = []
    for leg, m in zip(legs, masses):
        detour = leg.unknown * (1 + rng.exponential(0.5, size=samples))
        cells = leg.known + detour
        cost = cost + cells * per_cell + scan_cost
        turns = turns + cells + scan_turns
        arrive = t0 + turns
        p_batt = float(np.mean(battery0 - cost >= 0))
        p_alive = float(np.mean(life.p_alive(arrive)))
        keep = float(np.mean(retained(arrive - t_now, p_move)))
        term = (max(m * keep, 0) ** wl) * (p_alive ** wa) * (p_batt ** wb) * p_confirm
        total += term
        detail.append({"mass": round(m, 4), "retained": round(keep, 3), "p_alive": round(p_alive, 3),
                       "p_battery": round(p_batt, 3), "arrival": round(float(np.mean(arrive)), 1)})
    left = float(np.mean(np.maximum(battery0 - cost, 0)))
    return total, left, detail


def evaluate(t: int, battery: float, masses: list[float], dash_legs: list[Leg], safe_home: float,
             safe_legs: list[Leg], costs: dict, life, rng: np.random.Generator, params: dict,
             reserve: float, per_cell_return: float, p_move: float = 0.0, p_confirm: float = 0.85) -> dict:
    samples = int(params.get("samples", 30))
    w = (params.get("w_location", 1.0), params.get("w_alive", 1.0), params.get("w_battery", 1.0))
    per_cell = costs["move"] + 0.5 * costs["turn_per_90"]
    full_scan = costs["thermal"] + costs["cv"] + costs["turn"]
    lean_scan = costs["cv"] + costs["turn"]
    common = dict(per_cell=per_cell, life=life, rng=rng, samples=samples, w=w, p_move=p_move, t_now=t,
                  p_confirm=p_confirm)
    dash, dash_left, dash_d = _tour(battery, t, dash_legs, masses, full_scan, 3, **common)
    lean, lean_left, lean_d = _tour(battery, t, dash_legs, masses, lean_scan, 2, **common)
    # safe: walk home, one charge turn to 100%, then the tour from the hearth
    safe, safe_left, safe_d = _tour(100.0, t + safe_home + 1, safe_legs, masses, full_scan, 3, **common)
    # going home also keeps the option of searching again with a full battery: worth roughly the
    # mission's base success rate, if she is still alive by the time the agent is back out
    restart = float(params.get("restart_value", 0.5)) * float(life.p_alive(t + 2 * safe_home + 1 + 40))
    safe = max(safe, restart)
    if battery - per_cell_return * safe_home < 0:  # cannot even get home
        safe = 0.0
    plans = {"safe": round(safe, 4), "dash": round(dash, 4), "lean": round(lean, 4)}
    left = {"safe": safe_left, "dash": dash_left, "lean": lean_left}
    risky = "dash" if (dash > lean + 1e-9) or (abs(dash - lean) <= 1e-9 and dash_left > lean_left) else "lean"
    hyst = params.get("hysteresis", 0.10)
    choice = risky if plans[risky] >= plans["safe"] + hyst else "safe"
    return {"t": t, "plans": plans, "choice": choice, "battery_left": {k: round(v, 2) for k, v in left.items()},
            "p_chosen": plans[choice], "detail": {"safe": safe_d, "dash": dash_d, "lean": lean_d}}
