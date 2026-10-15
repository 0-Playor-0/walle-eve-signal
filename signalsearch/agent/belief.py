"""Belief map P(target in cell) over the agent frame.

Update rule: multiply-and-normalise (Bayes with per-cell likelihood ratios inside each scan's coverage;
outside coverage LR = 1), with each LR tempered (raised to a power < 1). A tiny uniform floor is mixed in after every update so that a wrong sensor
assumption can never make the true cell permanently impossible.
"""
from __future__ import annotations

import math

import numpy as np

from ..common.geometry import angle_diff
from .occupancy import FREE, UNKNOWN

FLOOR = 1e-3   # probability mass re-spread uniformly over possible cells after each update


def fft_convolve(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Same-size 2-D convolution (kernel centred), zero padded."""
    s0, s1 = a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1
    f = np.fft.irfft2(np.fft.rfft2(a, (s0, s1)) * np.fft.rfft2(k, (s0, s1)), (s0, s1))
    o0, o1 = k.shape[0] // 2, k.shape[1] // 2
    return f[o0:o0 + a.shape[0], o1:o1 + a.shape[1]]


def gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (x[:, None] ** 2 + x[None, :] ** 2) / max(sigma, 1e-6) ** 2)
    return g / g.sum()


class Belief:
    def __init__(self, size: int):
        self.p = np.full((size, size), 1.0 / (size * size))

    # ---- initialisation from the distress call -------------------------------------------------
    def init_from_signal(self, origin, signal: dict, sig_model: dict, p_move: float, possible: np.ndarray) -> None:
        n = self.p.shape[0]
        gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dx, dy = gx - origin[0], gy - origin[1]
        d = np.hypot(dx, dy)
        bearing = np.degrees(np.arctan2(dx, -dy)) % 360
        B = sig_model["bearing_noise_deg"]
        da = np.abs(angle_diff(bearing, signal["bearing_deg"]))
        ang = np.where(da <= B, 1.0, np.exp(-0.5 * ((da - B) / 6.0) ** 2))
        s_true = 1.0 / (1.0 + (d / sig_model["d0"]) ** 2)
        sd = sig_model["strength_noise"] * 1.3  # a little wider than the nominal noise
        s = min(max(signal["strength"], 1e-4), 0.9999)
        rad = np.exp(-0.5 * ((np.log(s) - np.log(np.maximum(s_true, 1e-9))) / sd) ** 2)
        lik = ang * rad
        lik[tuple(origin)] = 0.0
        # widen by how far the target may have wandered since the call
        spread = math.sqrt(max(signal.get("age", 0), 0) * p_move)
        if spread > 0.5:
            lik = np.maximum(fft_convolve(lik, gaussian_kernel(spread)), 0)
        lik = lik / lik.max()
        self.p = (lik + 1e-3) * possible
        self.p /= self.p.sum()

    # ---- prediction: the target may have moved -------------------------------------------------
    def predict(self, p_move: float, walkable: np.ndarray) -> None:
        if p_move <= 0:
            return
        w = walkable.astype(np.float64)
        deg = np.zeros_like(w)
        deg[1:, :] += w[:-1, :]
        deg[:-1, :] += w[1:, :]
        deg[:, 1:] += w[:, :-1]
        deg[:, :-1] += w[:, 1:]
        movable = (deg > 0) & walkable
        out = np.where(movable, p_move * self.p / np.maximum(deg, 1), 0.0)
        new = self.p - np.where(movable, p_move * self.p, 0.0)
        inflow = np.zeros_like(new)
        inflow[1:, :] += out[:-1, :]
        inflow[:-1, :] += out[1:, :]
        inflow[:, 1:] += out[:, :-1]
        inflow[:, :-1] += out[:, 1:]
        self.p = new + inflow * w

    # ---- measurement update --------------------------------------------------------------------
    def apply(self, cells: np.ndarray, lr: np.ndarray, temper: float = 1.0) -> None:
        """Multiply covered cells by LR**temper. temper < 1 discounts each reading for model mismatch and
        correlated errors (a moving target, shape confusion, persistent noise) so no single scan is final."""
        if len(cells) == 0:
            return
        self.p[cells[:, 0], cells[:, 1]] *= np.maximum(lr, 0.0) ** temper

    def zero_cells(self, cells) -> None:
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        if len(cells):
            self.p[cells[:, 0], cells[:, 1]] = 0.0

    def normalise(self, possible: np.ndarray) -> None:
        self.p *= possible
        tot = self.p.sum()
        n_possible = possible.sum()
        if tot <= 0 or not np.isfinite(tot):
            self.p = possible / max(n_possible, 1)
            return
        self.p = (1 - FLOOR) * self.p / tot + FLOOR * possible / max(n_possible, 1)

    # ---- summaries -----------------------------------------------------------------------------
    def mass_near(self, c, r: int = 2) -> float:
        x, y = c
        return float(self.p[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1].sum())

    def smoothed(self, r: int = 2) -> np.ndarray:
        k = np.ones((2 * r + 1, 2 * r + 1))
        return fft_convolve(self.p, k)

    def peaks(self, k: int = 3, r: int = 2, min_sep: int = 6) -> list[tuple[tuple[int, int], float]]:
        """Top-k cluster centres (by mass within radius r), at least min_sep apart."""
        sm = self.smoothed(r)
        order = np.argsort(sm, axis=None)[::-1]
        out: list[tuple[tuple[int, int], float]] = []
        for idx in order[:5000]:
            c = np.unravel_index(idx, sm.shape)
            c = (int(c[0]), int(c[1]))
            # far enough apart that the (2r+1)^2 mass boxes never overlap (masses then sum to <= 1)
            if all(abs(c[0] - o[0][0]) + abs(c[1] - o[0][1]) >= min_sep and
                   max(abs(c[0] - o[0][0]), abs(c[1] - o[0][1])) > 2 * r for o in out):
                # snap to the most likely cell of the cluster
                x0, y0 = max(0, c[0] - r), max(0, c[1] - r)
                sub = self.p[x0:c[0] + r + 1, y0:c[1] + r + 1]
                j = np.unravel_index(int(np.argmax(sub)), sub.shape)
                out.append(((x0 + int(j[0]), y0 + int(j[1])), float(sm[c])))
                if len(out) >= k:
                    break
        return out

    def entropy_bits(self) -> float:
        q = self.p[self.p > 0]
        return float(-(q * np.log2(q)).sum())

    def summary(self) -> dict:
        flat = np.argsort(self.p, axis=None)[::-1][:3]
        top = [[int(i // self.p.shape[1]), int(i % self.p.shape[1]), round(float(self.p.flat[i]), 5)] for i in flat]
        return {"peak": top[0][:2], "peak_prob": top[0][2], "entropy": round(self.entropy_bits(), 3), "top3": top}


def possible_mask(grid: np.ndarray) -> np.ndarray:
    """Cells where the target could be: free or unknown (not wall, not out of bounds)."""
    return (grid == FREE) | (grid == UNKNOWN)

