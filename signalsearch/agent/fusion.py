"""Detection fusion and source tracking.

Only one sensor fires per turn, so "simultaneous" readings are taken as those within a short window
(FUSE_WINDOW turns): a new detection whose position region overlaps an existing track (allowing for
the motion possible since) is merged into it with an inverse-variance position update. Each track
accumulates a signature (warm / heartbeat / LIDAR shape / CV label) and a speed estimate.

The belief map is updated from the raw per-sensor likelihoods (sensors_assumed) so evidence is never
counted twice; tracks drive identity reasoning, the motion estimate, decoy explanation and the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FUSE_WINDOW = 6
MAX_SPEED = 0.8  # cells/turn an animal can plausibly cover (for gating)


@dataclass
class Track:
    id: int
    pos: list[float]
    sigma: float
    first_t: int
    last_t: int
    hits: dict = field(default_factory=dict)   # e.g. {"thermal": 2, "radar": 1, "humanoid": 1}
    labels: dict = field(default_factory=dict)  # CV label counts
    history: list = field(default_factory=list)  # (t, x, y, sigma, sensor)
    speed: float = 0.0
    strength_max: float = 0.0

    @property
    def kind(self) -> str:
        """Identity: only a CV label is proof; otherwise a hypothesis from the signature."""
        if self.labels:
            lab = max(self.labels, key=self.labels.get)
            return {"target": "target", "dog": "dog", "fire": "torch", "hearth": "hearth", "armour": "armour"}[lab]
        return "unknown"

    def guess(self) -> str:
        if self.labels:
            return self.kind
        warm, beat = self.hits.get("thermal", 0), self.hits.get("radar", 0)
        if self.hits.get("box"):
            return "hearth?"
        if self.strength_max > 0.6:
            return "fire?"
        if beat and self.hits.get("humanoid"):
            return "target?"
        if beat and self.hits.get("small"):
            return "dog?"
        if beat or warm:
            return "living?" if beat else "warm source?"
        if self.hits.get("humanoid"):
            return "humanoid (armour or target)?"
        return "unknown"

    def to_dict(self) -> dict:
        return {"id": self.id, "pos": [round(self.pos[0], 2), round(self.pos[1], 2)], "sigma": round(self.sigma, 2),
                "last_t": self.last_t, "hits": dict(self.hits), "labels": dict(self.labels), "guess": self.guess(),
                "speed": round(self.speed, 3)}


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self._next = 0

    def _gate(self, tr: Track, pos, sigma: float, t: int) -> float:
        dt = t - tr.last_t
        moving = 0.0 if tr.kind in ("torch", "hearth", "armour") else MAX_SPEED * dt
        return math.dist(tr.pos, pos) - (2 * (tr.sigma + sigma) + min(moving, 6) + 1)

    def add(self, t: int, sensor: str, pos, sigma: float, attr: str | None = None, label: str | None = None,
            strength: float = 0.0) -> Track:
        best, best_g = None, 0.0
        for tr in self.tracks:
            if t - tr.last_t > 60:
                continue
            g = self._gate(tr, pos, sigma, t)
            if g <= 0 and (best is None or g < best_g):
                best, best_g = tr, g
        if best is None:
            best = Track(self._next, [float(pos[0]), float(pos[1])], sigma, t, t)
            self._next += 1
            self.tracks.append(best)
        else:
            dt = t - best.last_t
            prior_sigma = best.sigma + (0 if best.kind in ("torch", "hearth", "armour") else 0.5 * dt)
            w1, w2 = 1 / prior_sigma ** 2, 1 / max(sigma, 0.3) ** 2
            old = list(best.pos)
            best.pos = [(old[0] * w1 + pos[0] * w2) / (w1 + w2), (old[1] * w1 + pos[1] * w2) / (w1 + w2)]
            best.sigma = max(0.3, math.sqrt(1 / (w1 + w2)))
            if dt > 0 and sigma <= 1.0:
                inst = math.dist(old, pos) / dt
                best.speed = 0.7 * best.speed + 0.3 * inst
        best.last_t = t
        best.hits[sensor if attr is None else attr] = best.hits.get(sensor if attr is None else attr, 0) + 1
        if attr is not None and sensor != attr:
            best.hits[sensor] = best.hits.get(sensor, 0) + 1
        if label:
            best.labels[label] = best.labels.get(label, 0) + 1
        best.strength_max = max(best.strength_max, strength)
        best.history.append((t, float(pos[0]), float(pos[1]), sigma, sensor))
        return best

    def recent(self, t: int, window: int = FUSE_WINDOW) -> list[Track]:
        return [tr for tr in self.tracks if t - tr.last_t <= window]

    def known_decoys(self, t: int) -> list[dict]:
        """Identified (CV-labelled) non-target sources, for clutter explanation in the sensor models.
        Static decoys stay known forever; a dog only while its last sighting is fresh."""
        out = []
        for tr in self.tracks:
            if tr.kind in ("unknown", "target") or (tr.kind == "dog" and t - tr.last_t > 10):
                continue
            sigma = tr.sigma + (MAX_SPEED * (t - tr.last_t) if tr.kind == "dog" else 0.0)
            out.append({"kind": tr.kind, "pos": [round(tr.pos[0]), round(tr.pos[1])], "sigma": sigma})
        return out

    def sources(self, t: int, max_age: int = 40) -> list[dict]:
        """Unidentified sources already seen. Fed to the sensor models as clutter that is known to exist,
        so seeing the same source again only weighs target-vs-decoy signatures, never "something vs nothing"."""
        out = []
        for tr in self.tracks:
            if tr.kind != "unknown" or t - tr.last_t > max_age:
                continue
            living = tr.hits.get("radar", 0) > 0
            sigma = tr.sigma + (0.3 * (t - tr.last_t) if living else 0.0)
            out.append({"kind": "source", "pos": [tr.pos[0], tr.pos[1]], "sigma": sigma, "hits": dict(tr.hits)})
        return out

