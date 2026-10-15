"""The agent's sensor models (built from agent_assumed only).

Each `*_likelihood` returns (cells, LR): the cells inside this scan's coverage and, for each, the ratio
    LR(c) = P(reading | target in c) / P(reading | target not in coverage)
Cells outside coverage are simply not returned (LR = 1). For blob sensors the standard single-target
clutter model is used:
    LR(c) = (1 - p_det(c)) + sum_j  density_target(blob_j | c) / density_clutter(blob_j)
where clutter density comes from the assumed signature table (how many dogs/torches/armour/hearths a
castle holds, and what each one does to each sensor) plus decoys the agent has already identified.
"""
from __future__ import annotations

import math

import numpy as np

from ..common.geometry import disk_cells, ray_profile

FREE_FRACTION = 0.65      # a-priori: roughly two thirds of a castle's cells are floor
WALL_DENSITY = 1 / 14.0   # expected walls per metre of unexplored castle along a ray
MODEL_SD = 0.35           # relative uncertainty of the agent's own expected-strength model


def phi(x):
    """Standard normal CDF (tanh approximation, vectorised)."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def npdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


class SensorModels:
    def __init__(self, assumed: dict):
        self.a = assumed
        self.s = assumed["sensors"]
        n = assumed["map_size"]
        self.n_free = FREE_FRACTION * n * n
        cnt = assumed["signature_counts"]
        self.rho = {k: float(v) / self.n_free for k, v in cnt.items()}
        self.weather = {"temperature": 25.0, "humidity": 60.0, "rain": 0.0}

    # ---- weather-dependent quantities ------------------------------------------------------------
    def set_weather(self, w: dict) -> None:
        self.weather = dict(w)

    def lidar_range(self) -> float:
        sp, w = self.s["lidar"], self.weather
        return sp["range"] * max(0.3, 1 - sp["humidity_loss"] * w["humidity"] - sp["rain_loss"] * w["rain"])

    def intensity(self, kind: str) -> float:
        temps = self.a["temps"]
        amb = self.weather["temperature"]
        t = {"target": temps["human"], "dog": temps["dog"], "torch": temps["torch"], "hearth": temps["hearth"]}.get(kind)
        if t is None:
            return 0.0
        c = abs(t - amb)
        return c / (c + self.s["thermal"]["contrast_ref"])

    def thermal_expected(self, kind: str, d, walls_est):
        sp, w = self.s["thermal"], self.weather
        att = sp["humidity_att"] * w["humidity"] + sp["rain_att"] * w["rain"]
        d = np.asarray(d, dtype=np.float64)
        return self.intensity(kind) / (1 + (d / sp["d0"]) ** 2) * sp["wall_factor"] ** walls_est * np.exp(-att * d)

    def thermal_sd(self, mu):
        return np.sqrt(self.s["thermal"]["noise_sd"] ** 2 + (MODEL_SD * mu) ** 2)

    def _strength_lik(self, s, mu, sd):
        """Density of observed strength s; the sensor saturates at 1.0 so treat s=1 as a tail mass."""
        if s >= 0.999:
            return 1.0 - phi((0.999 - mu) / sd)
        return npdf(s, mu, sd)

    def _decoy_mix(self, src: dict) -> dict:
        """If a seen source is not the target, what is it? Prior counts, restricted by its evidence."""
        h = src.get("hits", {})
        if h.get("radar"):
            return {"dog": 1.0}
        kinds = ["dog", "torch", "hearth"] if h.get("thermal") else ["dog", "torch", "hearth", "armour"]
        if h.get("humanoid"):
            kinds = [k for k in kinds if k in ("armour", "dog", "torch")]
        tot = sum(self.rho[k] for k in kinds)
        return {k: self.rho[k] / tot for k in kinds}

    # ---- coverage helpers ------------------------------------------------------------------------
    def _coverage(self, pos, r, occ, feasible):
        cells = disk_cells(pos[0], pos[1], r, occ.grid.shape)
        ok = feasible[cells[:, 0], cells[:, 1]] & ~occ.wall[cells[:, 0], cells[:, 1]]
        return cells[ok]

    def _walls_est(self, occ, pos, cells):
        walls, unk = ray_profile(occ.wall, occ.unknown, pos, cells)
        return walls + unk * WALL_DENSITY

    # ---- thermal ---------------------------------------------------------------------------------
    def thermal_likelihood(self, pos, blobs, occ, feasible, known: list[dict]):
        sp = self.s["thermal"]
        cells = self._coverage(pos, sp["range"], occ, feasible)
        if len(cells) == 0:
            return cells, np.ones(0)
        d = np.hypot(cells[:, 0] - pos[0], cells[:, 1] - pos[1])
        west = self._walls_est(occ, pos, cells)
        mu = self.thermal_expected("target", d, west)
        sd = self.thermal_sd(mu)
        p_det = phi((mu - sp["threshold"]) / sd)
        lr = 1.0 - p_det
        if blobs:
            bpos = np.array([b["pos"] for b in blobs], dtype=np.float64)
            bd = np.hypot(bpos[:, 0] - pos[0], bpos[:, 1] - pos[1])
            bwest = self._walls_est(occ, pos, bpos)
            for j, b in enumerate(blobs):
                sigma = sp["blur_base"] + bd[j] * sp["blur_per_m"] + 0.5
                g = np.exp(-0.5 * ((cells[:, 0] - bpos[j, 0]) ** 2 + (cells[:, 1] - bpos[j, 1]) ** 2) / sigma ** 2) \
                    / (2 * math.pi * sigma ** 2)
                s = float(b["strength"])
                tgt = g * self._strength_lik(s, mu, sd)
                lam = 1e-6
                for kind in ("dog", "torch", "hearth"):
                    mk = float(self.thermal_expected(kind, bd[j], bwest[j]))
                    lam += self.rho[kind] * float(self._strength_lik(s, mk, float(self.thermal_sd(mk))))
                for k in known:  # identified decoys and already-seen sources explain blobs near them
                    if k["kind"] == "source" and k["hits"].get("thermal"):
                        dd2 = (k["pos"][0] - bpos[j, 0]) ** 2 + (k["pos"][1] - bpos[j, 1]) ** 2
                        sk = sigma + k["sigma"]
                        mix = self._decoy_mix(k)
                        lik = sum(w * float(self._strength_lik(s, mk, float(self.thermal_sd(mk))))
                                  for kind, w in mix.items() if kind != "armour"
                                  for mk in [float(self.thermal_expected(kind, bd[j], bwest[j]))])
                        lam += math.exp(-0.5 * dd2 / sk ** 2) / (2 * math.pi * sk ** 2) * lik
                    if k["kind"] in ("dog", "torch", "hearth"):
                        dd2 = (k["pos"][0] - bpos[j, 0]) ** 2 + (k["pos"][1] - bpos[j, 1]) ** 2
                        sk = sigma + k.get("sigma", 0.0)
                        mk = float(self.thermal_expected(k["kind"], bd[j], bwest[j]))
                        lam += math.exp(-0.5 * dd2 / sk ** 2) / (2 * math.pi * sk ** 2) * \
                            float(self._strength_lik(s, mk, float(self.thermal_sd(mk))))
                lr = lr + tgt / lam
        return cells, lr

    # ---- radar -----------------------------------------------------------------------------------
    def radar_pdet(self, walls_est):
        sp = self.s["radar"]
        return sp["p_detect"] * sp["wall_atten"] ** walls_est

    def radar_likelihood(self, pos, blobs, occ, feasible, known: list[dict]):
        sp = self.s["radar"]
        cells = self._coverage(pos, sp["range"], occ, feasible)
        if len(cells) == 0:
            return cells, np.ones(0)
        p = self.radar_pdet(self._walls_est(occ, pos, cells))
        lr = 1.0 - p
        n = int(sp["pos_noise"])
        box = 1.0 / (2 * n + 1) ** 2
        area = math.pi * sp["range"] ** 2
        for b in blobs:
            z = b["pos"]
            near = (np.abs(cells[:, 0] - z[0]) <= n) & (np.abs(cells[:, 1] - z[1]) <= n)
            g = np.where(near, box, box * 1e-3)
            lam = self.rho["dog"] * sp["p_detect"] + sp["false_alarm"] / area + 1e-7
            for k in known:
                living = k["kind"] == "dog" or (k["kind"] == "source" and k["hits"].get("radar"))
                reach = n + 2 + k.get("sigma", 0.0)
                if living and abs(k["pos"][0] - z[0]) <= reach and abs(k["pos"][1] - z[1]) <= reach:
                    lam += box * sp["p_detect"]
            lr = lr + p * g / lam
        return cells, lr

    # ---- lidar -----------------------------------------------------------------------------------
    def lidar_likelihood(self, free_cells, blobs, feasible, known: list[dict]):
        sp = self.s["lidar"]
        cells = np.asarray(free_cells, dtype=np.int64).reshape(-1, 2)
        if len(cells) == 0:
            return cells, np.ones(0)
        cells = cells[feasible[cells[:, 0], cells[:, 1]]]
        ps, conf = sp["p_shape"], sp["shape_confusion"]
        lr = np.full(len(cells), 1.0 - ps)
        index = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
        known_at = {tuple(k["pos"]): k["kind"] for k in known if k.get("sigma", 9) <= 0.5}
        r = self.rho
        clutter = {"humanoid": r["armour"] * ps * (1 - conf) + (r["dog"] + r["torch"]) * ps * conf,
                   "small": (r["dog"] + r["torch"]) * ps * (1 - conf) + r["armour"] * ps * conf,
                   "box": r["hearth"] * ps}
        target_p = {"humanoid": ps * (1 - conf), "small": ps * conf, "box": 1e-3}
        shape_of = {"dog": "small", "torch": "small", "armour": "humanoid", "hearth": "box"}
        sources = [k for k in known if k["kind"] == "source"]
        for b in blobs:
            i = index.get(tuple(b["pos"]))
            if i is None:
                continue
            lam = clutter[b["shape"]] + 1e-6
            if tuple(b["pos"]) in known_at:  # already identified decoy standing there
                lam += ps
            for k in sources:  # a source already seen here: which decoy would give this shape?
                if math.dist(k["pos"], b["pos"]) <= 1.5 + k["sigma"]:
                    for kind, w in self._decoy_mix(k).items():
                        own = shape_of[kind]
                        if own == b["shape"]:
                            lam += w * ps * (1 - conf if own in ("humanoid", "small") else 1.0)
                        elif {own, b["shape"]} == {"humanoid", "small"}:
                            lam += w * ps * conf
                    break
            lr[i] = target_p[b["shape"]] / lam
        return cells, lr

    # ---- cv --------------------------------------------------------------------------------------
    def cv_likelihood(self, visible, labels, feasible, known: list[dict]):
        sp = self.s["cv"]
        cells = np.asarray(visible, dtype=np.int64).reshape(-1, 2)
        if len(cells) == 0:
            return cells, np.ones(0)
        cells = cells[feasible[cells[:, 0], cells[:, 1]]]
        pd = sp["p_detect"]
        lr = np.full(len(cells), 1.0 - pd)
        index = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
        armour_at = {tuple(k["pos"]) for k in known if k["kind"] == "armour"}
        for lab in labels:
            i = index.get(tuple(lab["pos"]))
            if i is None:
                continue
            if lab["label"] == "target":
                fp = self.rho["armour"] * sp["armour_false_target"] + 1e-4
                if tuple(lab["pos"]) in armour_at:
                    fp += sp["armour_false_target"]
                lr[i] = pd / fp
            else:
                lr[i] = min(lr[i], 1.0 - pd)
        return cells, lr
