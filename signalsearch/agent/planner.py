"""Where to go and what to scan.

For every candidate scan position p and sensor s the expected detection mass is
    q_s(p) = sum_c belief(c) * p_det_s(|c - p|)
computed for all p at once as an FFT convolution of the belief with a detection kernel (walls along
the way approximated by the castle's expected wall density). The information gain of the scan is the
entropy of its detect / no-detect outcome, H(min(q, 0.5)) bits.
  position:  argmax_p  max_s  q_s(p) / (travel_cost(p) + cost_s)          (the spec's mass / cost score)
  sensor:    argmax_s  H(q_s(p)) / (travel_cost(p) + cost_s) at the chosen position   (gain per % spent)
both subject to   battery - travel - cost_s - per_cell * h_home(p) - reserve >= 0.
"""
from __future__ import annotations

import math

import numpy as np

from .belief import fft_convolve
from .sensors_assumed import WALL_DENSITY, SensorModels, phi

LIDAR_VISIBILITY = 5.0  # metres: expected line-of-sight decay inside rooms


def hb(q):
    q = np.clip(q, 1e-9, 0.5)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def _radial(r: float, f) -> np.ndarray:
    ri = int(math.floor(r))
    x = np.arange(-ri, ri + 1)
    d = np.hypot(x[:, None], x[None, :])
    return np.where(d <= r, f(d), 0.0)


class Planner:
    def __init__(self, models: SensorModels, costs: dict, params: dict):
        self.m = models
        self.costs = costs
        self.p = params
        self._kernels = None

    def kernels(self) -> dict:
        if self._kernels is None:
            m, s = self.m, self.m.s

            def thermal(d):
                mu = m.thermal_expected("target", d, d * WALL_DENSITY)
                return phi((mu - s["thermal"]["threshold"]) / m.thermal_sd(mu))

            self._kernels = {
                "thermal": _radial(s["thermal"]["range"], thermal),
                "radar": _radial(s["radar"]["range"], lambda d: m.radar_pdet(d * WALL_DENSITY)),
                "lidar": _radial(m.lidar_range(), lambda d: s["lidar"]["p_shape"] * (1 - s["lidar"]["shape_confusion"])
                                 * np.exp(-d / LIDAR_VISIBILITY)),
            }
        return self._kernels

    def reset_kernels(self) -> None:
        self._kernels = None

    def choose(self, belief: np.ndarray, window, explore: np.ndarray, home: np.ndarray, h: float, battery: float,
               pos, scans: dict, t: int, reserve: float, per_cell_return: float, extra_candidates=(),
               ignore_battery: bool = False, sensors=("thermal", "radar", "lidar")) -> dict | None:
        """Scan position by q(p) / (travel + scan cost); sensor there by information gain per % spent."""
        x0, x1, y0, y1 = window
        b = belief[x0:x1 + 1, y0:y1 + 1]
        ex = explore[x0:x1 + 1, y0:y1 + 1]
        hm = home[x0:x1 + 1, y0:y1 + 1]
        stride = int(self.p["candidate_stride"])
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing="ij")
        reach = np.isfinite(ex) & (ex >= 0)
        cand = reach & (gx % stride == 0) & (gy % stride == 0)
        for c in list(extra_candidates) + [tuple(pos)]:
            cx, cy = c[0] - x0, c[1] - y0
            if 0 <= cx < b.shape[0] and 0 <= cy < b.shape[1] and reach[cx, cy]:
                cand[cx, cy] = True
        idx = np.argwhere(cand)
        if len(idx) == 0:
            return None
        dist = ex[idx[:, 0], idx[:, 1]].astype(np.float64)
        travel = dist * (self.costs["move"] + 0.5 * self.costs["turn_per_90"])
        hv = hm[idx[:, 0], idx[:, 1]].astype(np.float64)
        home_bound = np.where(hv >= 0, np.minimum(hv, h + dist), h + dist)
        abs_x, abs_y = idx[:, 0] + x0, idx[:, 1] + y0
        need_home = per_cell_return * home_bound + reserve
        qs, rep, pos_score = {}, {}, np.full(len(idx), -1.0)
        ok_any = np.zeros(len(idx), dtype=bool)
        for s in sensors:
            q = fft_convolve(b, self.kernels()[s])[idx[:, 0], idx[:, 1]]
            rep[s] = self._repeat_factor(s, abs_x, abs_y, scans, t)
            qs[s] = q
            cost = self.costs[s]
            ok = np.ones(len(idx), dtype=bool) if ignore_battery else (battery - travel - cost - need_home >= 0)
            ok_any |= ok
            combined = travel + cost
            pos_score = np.where(ok, np.maximum(pos_score, q * rep[s] / combined), pos_score)
        if not ok_any.any():
            return None
        i = int(np.argmax(pos_score))
        # sensor at the chosen position: highest expected information gain per % spent getting there and
        # scanning (scan cost alone would always pick 0.2% LIDAR after a long walk)
        best_s, best_v = None, -1.0
        for s in sensors:
            cost = self.costs[s]
            if not ignore_battery and battery - travel[i] - cost - need_home[i] < 0:
                continue
            v = float(hb(qs[s][i]) * rep[s][i] / (travel[i] + cost))
            if v > best_v:
                best_s, best_v = s, v
        s = best_s
        rate = float(hb(qs[s][i]) * rep[s][i] / (travel[i] + self.costs[s]))
        top = np.argsort(-pos_score)[:5]
        candidates = [{"goal": [int(abs_x[j]), int(abs_y[j])], "score": round(float(pos_score[j]), 4),
                       "dist": round(float(dist[j]), 1)} for j in top if pos_score[j] >= 0]
        return {"goal": (int(abs_x[i]), int(abs_y[i])), "sensor": s, "score": float(pos_score[i]),
                "candidates": candidates, "rate": rate,
                "gain": float(hb(qs[s][i]) * rep[s][i]), "q": float(qs[s][i]), "travel": float(travel[i]),
                "dist": round(float(dist[i]), 1), "needed": float(travel[i] + self.costs[s] + need_home[i]),
                "home_bound": float(home_bound[i])}

    def _repeat_factor(self, s: str, xs, ys, scans: dict, t: int) -> np.ndarray:
        if not self.p.get("repeat_penalties", True):
            return np.ones(len(xs))
        rule = {"thermal": (self.p["thermal_repeat_radius"], 80, 0.02), "radar": (3, 12, 0.1),
                "lidar": (1.5, 6, 0.01)}[s]
        radius, age, factor = rule
        f = np.ones(len(xs))
        for st, (sx, sy) in scans.get(s, []):
            if t - st <= age:
                f = np.where((xs - sx) ** 2 + (ys - sy) ** 2 <= radius ** 2, factor, f)
        return f


    def gain_here(self, belief: np.ndarray, pos, sensor: str) -> float:
        """Information (bits) a scan from `pos` is expected to yield about the target."""
        k = self.kernels()[sensor]
        r = k.shape[0] // 2
        x, y = pos
        n0, n1 = belief.shape
        x0, x1, y0, y1 = max(0, x - r), min(n0, x + r + 1), max(0, y - r), min(n1, y + r + 1)
        sub = belief[x0:x1, y0:y1]
        ks = k[x0 - (x - r):x1 - (x - r), y0 - (y - r):y1 - (y - r)]
        return float(hb((sub * ks).sum()))
