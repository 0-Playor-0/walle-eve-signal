"""Target movement-rate estimate (Beta posterior on "moved this turn") and remaining-life distribution.

The agent knows the *form* of the target model: p_move = p0 * E, E falls roughly linearly to 0 over the
target's total life L. Its prior over L uses the same fairness rule the mission designer uses, applied to
the agent's own distance estimate from the distress call, with a lognormal spread.
"""
from __future__ import annotations

import math

import numpy as np

N_LIFE_SAMPLES = 400
PRIOR_STRENGTH = 8.0   # pseudo-turns of the Beta prior
PATH_FACTOR = 1.4      # median walking / straight-line distance in castles of this family (a-priori)


def _normal_quantiles(n: int) -> np.ndarray:
    # deterministic standard-normal quantiles via the inverse-erf series (Giles approximation)
    u = (np.arange(n) + 0.5) / n
    x = 2 * u - 1
    w = -np.log((1 - x) * (1 + x))
    small = w < 5
    ws = w - 2.5
    p1 = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087, -0.00125372503,
              -0.00417768164, 0.246640727, 1.50140941):
        p1 = c + p1 * ws
    wl = np.sqrt(np.where(small, 5, w)) - 3
    p2 = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773, -0.0076224613,
              0.00943887047, 1.00167406, 2.83297682):
        p2 = c + p2 * wl
    return math.sqrt(2) * np.where(small, p1, p2) * x


def life_prior_mean(distance_est: float, k: float) -> float:
    d_path = PATH_FACTOR * distance_est
    est_battery = 0.2 * 1.8 * d_path + 60
    allowance = 15 * max(0, math.ceil(est_battery / 100) - 1)
    return k * (1.5 * 1.2 * d_path + 10 + allowance)


class LifeModel:
    def __init__(self, mean_life: float, cv: float):
        sigma = math.sqrt(math.log(1 + cv ** 2))
        mu = math.log(max(mean_life, 10.0)) - sigma ** 2 / 2
        self.L = np.exp(mu + sigma * _normal_quantiles(N_LIFE_SAMPLES))
        self.logw = np.zeros(N_LIFE_SAMPLES)

    @property
    def w(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def energy(self, t: float) -> np.ndarray:
        return np.clip(1.0 - t / self.L, 0.0, 1.0)

    def mean_energy(self, t: float) -> float:
        return float((self.w * self.energy(t)).sum())

    def observe(self, t: float, moves: int, turns: int, p0: float) -> None:
        p = np.clip(p0 * self.energy(t), 1e-3, 1 - 1e-3)
        self.logw += moves * np.log(p) + (turns - moves) * np.log(1 - p)

    def p_alive(self, t_abs) -> np.ndarray | float:
        t_abs = np.asarray(t_abs, dtype=np.float64)
        return (self.w[None, :] * (self.L[None, :] > t_abs.reshape(-1, 1))).sum(axis=1).reshape(t_abs.shape)

    def cdf(self, x: float, bw: float = 0.05) -> float:
        """P(L <= x), smoothed with a lognormal kernel of log-sd bw per sample, computed through erfc so that
        differences of tiny tail probabilities keep their precision."""
        if x <= 0:
            return 0.0
        lx = math.log(x)
        w = self.w
        return float(sum(wi * 0.5 * math.erfc(-(lx - math.log(li)) / (bw * math.sqrt(2)))
                         for wi, li in zip(w, self.L)))

    def remaining(self, t: float) -> dict:
        rem = np.maximum(self.L - t, 0.0)
        w = self.w
        order = np.argsort(rem)
        cw = np.cumsum(w[order])
        q = lambda f: float(rem[order][min(len(rem) - 1, int(np.searchsorted(cw, f)))])
        return {"mean": round(float((w * rem).sum()), 1), "p10": round(q(0.1), 1), "p50": round(q(0.5), 1),
                "p90": round(q(0.9), 1)}


class MotionEstimator:
    """Beta posterior on the target's per-turn move probability from precise re-sightings."""

    def __init__(self, p0: float, life: LifeModel, window: int = 80):
        self.p0 = p0
        self.life = life
        self.window = window
        self.pairs: list[tuple[int, int, int]] = []  # (t, moves, turns)
        self._seen: set[tuple[int, int]] = set()

    def add_track(self, track, t: int) -> None:
        """Use consecutive precise sightings (sigma <= 1) of a target-candidate track."""
        pts = [(ht, round(x), round(y)) for ht, x, y, s, _ in track.history if s <= 1.0]
        for (t1, x1, y1), (t2, x2, y2) in zip(pts, pts[1:]):
            key = (track.id, t2)
            if key in self._seen or t2 <= t1 or t2 - t1 > 30:
                continue
            self._seen.add(key)
            moves = min(abs(x2 - x1) + abs(y2 - y1), t2 - t1)
            self.pairs.append((t2, moves, t2 - t1))
            self.life.observe((t1 + t2) / 2, moves, t2 - t1, self.p0)

    def estimate(self, t: int) -> float:
        m0 = self.p0 * self.life.mean_energy(t)
        a = PRIOR_STRENGTH * m0
        b = PRIOR_STRENGTH * (1 - m0)
        for pt, moves, turns in self.pairs:
            if t - pt <= self.window:
                a += moves
                b += turns - moves
        return float(a / (a + b))

    def energy_estimate(self, t: int) -> float:
        return float(np.clip(self.estimate(t) / self.p0, 0.0, 1.0))
