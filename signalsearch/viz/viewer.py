"""Pygame viewer. Draws one FrameData dict (built from logs by replay.py or live by play.py):

  left   - the agent's mind: occupancy (unknown as fog, with a darker band at its edge), the belief as a cool
           glow with its 50% / 90% likely areas outlined and the peak marked, known hearths, planned path,
           a sonar ping for each scan (findings appear as the ring reaches them), fused detections, goal
  right  - ground truth: the castle with sprites, warm lights on torches and hearths, rescuer, target
  bottom - battery and life bars, mode, turn, the override banner (own row), reasoning, the event timeline
  far right - receipts: every battery "purchase", one colour per action type
All colours come from themes.PALETTE (warm hues are heat and fire only; the belief is cool).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from ..common.actions import DIRS
from .assets import Sprites, scaled
from .themes import PALETTE as PAL
from .themes import theme

W, H = 1280, 760
MIND = pygame.Rect(10, 64, 500, 500)
TRUTH = pygame.Rect(520, 64, 500, 500)
RECEIPT = pygame.Rect(1030, 64, 240, 500)
BAR = pygame.Rect(10, 576, 1260, 176)
TIMELINE = pygame.Rect(BAR.x + 12, BAR.bottom - 46, BAR.w - 24, 12)
CAM = 31  # truth camera: cells per side
ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"
FONTS = ASSETS / "fonts"
PING_SECONDS = 0.4
SCANS = ("lidar", "radar", "thermal", "cv")
MONO = "menlo,dejavusansmono,couriernew,monospace"


# ---- receipts and events -------------------------------------------------------------------------
def receipt(obs: dict) -> dict:
    """One battery 'purchase': {t, kind, act, cost, bal, res, flag}; flag is 'detect' or 'label' for findings."""
    a = obs["action"]
    r = obs.get("readings", {}) or {}
    kind = "spawn" if a == "spawn" else a.split()[0]
    res, flag = "", None
    if a == "spawn":
        res = "on a hearth"
    elif "lidar" in r:
        L = r["lidar"]
        shapes = [b for b in L["blobs"] if b["shape"] != "box"]
        res = f"{len(L['free'])} free, {len(L['walls'])} walls, {len(L['blobs'])} shapes"
        flag = "detect" if shapes else None
    elif "radar" in r:
        res = f"{len(r['radar'])} heartbeat{'s' if len(r['radar']) != 1 else ''}"
        flag = "detect" if r["radar"] else None
    elif "thermal" in r:
        res = f"{len(r['thermal'])} warm blob{'s' if len(r['thermal']) != 1 else ''}"
        flag = "detect" if r["thermal"] else None
    elif "cv" in r:
        labs = [x["label"] for x in r["cv"]["labels"]]
        res = ", ".join(labs) if labs else "nothing"
        flag = "label" if labs else None
    elif kind == "charge":
        res = f"+{obs.get('charged', 0):.0f}%"
    elif obs.get("blocked"):
        res = "blocked!"
    return {"t": obs["t"], "kind": kind, "act": a.upper(), "cost": float(obs.get("cost") or 0.0),
            "bal": float(obs["battery"]), "res": res, "flag": flag}


def events_of(obs: dict, override=None) -> list[str]:
    """Timeline markers for one turn: detections by sensor, camera labels, charges, overrides."""
    r = obs.get("readings", {}) or {}
    out = []
    if r.get("thermal"):
        out.append("thermal")
    if r.get("radar"):
        out.append("radar")
    if "lidar" in r and any(b["shape"] != "box" for b in r["lidar"]["blobs"]):
        out.append("lidar")
    if "cv" in r and r["cv"]["labels"]:
        out.append("cv")
    if (obs.get("action") or "").startswith("charge"):
        out.append("charge")
    if override:
        out.append("override")
    return out


def _font(name: str, size: int, fallback: str, bold: bool = False) -> pygame.font.Font:
    path = FONTS / name
    if path.exists():
        return pygame.font.Font(str(path), size)
    return pygame.font.SysFont(fallback, size, bold=bold)


def _edges(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cell-edge segments (x0, y0, x1, y1) of a boolean region's outline."""
    m = np.pad(mask, 1)
    v = np.argwhere(m[1:, 1:-1] != m[:-1, 1:-1])   # between column x-1 and x: vertical edge at x
    h = np.argwhere(m[1:-1, 1:] != m[1:-1, :-1])   # between row y-1 and y: horizontal edge at y
    return [(x, y, x, y + 1) for x, y in v] + [(x, y, x + 1, y) for x, y in h]


def _likely_threshold(p: np.ndarray, mass: float) -> float:
    """Smallest belief value whose top cells hold `mass` of the total (the likely area is p >= it)."""
    flat = np.sort(p.ravel())[::-1]
    c = np.cumsum(flat)
    i = int(np.searchsorted(c, mass * c[-1]))
    return float(flat[min(i, len(flat) - 1)])


class Viewer:
    def __init__(self, cfg: dict, walls: np.ndarray, static: dict, pack: str | None = None):
        self.cfg = cfg
        self.th = theme(cfg)
        self.walls = walls
        self.static = static
        self.offset = static["offset"]
        self.n = walls.shape[0]
        self.sprites = Sprites(ASSETS, pack)
        self.cache: dict = {}
        self.layers = {"belief": True, "fog": True, "path": True, "truth": True, "legend": False}
        self.fullmap = False
        self.static_ping = False          # GIF export / screenshots: ring at full radius, low opacity
        self.ping: dict | None = None     # {"key", "start"}: the scan being animated
        self.events: dict[int, list[str]] = {}
        self.n_turns = 0
        self._mind = None                 # cached mind layer for the current frame
        pygame.font.init()
        self.font = pygame.font.SysFont(MONO, 13)
        self.small = pygame.font.SysFont(MONO, 11)
        self.num = pygame.font.SysFont(MONO, 20, bold=True)
        self.title = _font("Silkscreen-Bold.ttf", 20, "avenirnext,helveticaneue,arial", bold=True)
        self.head = _font("Silkscreen-Regular.ttf", 12, "avenirnext,helveticaneue,arial", bold=True)
        self.huge = _font("Silkscreen-Bold.ttf", 36, "avenirnext,helveticaneue,arial", bold=True)
        sens = cfg["agent_assumed"]["sensors"]
        self.ranges = {"lidar": sens["lidar"]["range"], "radar": sens["radar"]["range"], "thermal": sens["thermal"]["range"]}
        self._castle = self._render_castle()
        self._castle_small = pygame.transform.smoothscale(self._castle, (TRUTH.w, TRUTH.h))
        self._fx = pygame.Surface((MIND.w, MIND.h), pygame.SRCALPHA)

    # ---- timeline ---------------------------------------------------------------------------------
    def set_timeline(self, events: dict[int, list[str]], n_turns: int) -> None:
        """Markers for the whole mission (replay) and its length."""
        self.events, self.n_turns = dict(events), n_turns

    def add_events(self, t: int, kinds: list[str]) -> None:
        """Append one turn's markers (live play, where the mission grows)."""
        if kinds:
            self.events.setdefault(t, []).extend(kinds)
        self.n_turns = max(self.n_turns, t + 1)

    def timeline_turn(self, pos) -> int | None:
        """The turn under a click on the timeline, or None."""
        if not TIMELINE.inflate(0, 10).collidepoint(pos) or self.n_turns <= 1:
            return None
        f = (pos[0] - TIMELINE.x) / TIMELINE.w
        return int(round(min(max(f, 0.0), 1.0) * (self.n_turns - 1)))

    # ---- static castle ------------------------------------------------------------------------
    def _render_castle(self) -> pygame.Surface:
        n, T = self.n, 16
        surf = pygame.Surface((n * T, n * T))
        dim = pygame.Surface((T, T))
        dim.fill(PAL["dim_floor"])
        for x in range(n):
            for y in range(n):
                if self.walls[x, y]:
                    surf.blit(scaled(self.sprites.get("wall"), T, self.cache), (x * T, y * T))
                else:
                    surf.blit(scaled(self.sprites.variant("floor", x, y), T, self.cache), (x * T, y * T))
                    surf.blit(dim, (x * T, y * T), special_flags=pygame.BLEND_MULT)
        shadow = pygame.Surface((T, 4), pygame.SRCALPHA)
        shadow.fill(PAL["wall_shadow"])
        for x, y in np.argwhere(self.walls[:, :-1] & ~self.walls[:, 1:]):  # wall face casts a shadow below
            surf.blit(shadow, (x * T, (y + 1) * T))
        if self.th["name"] == "walle":  # rusty tint for the junkyard skin
            tint = pygame.Surface(surf.get_size())
            tint.fill(self.th["tint_mult"])
            surf.blit(tint, (0, 0), special_flags=pygame.BLEND_MULT)
        return surf

    # ---- helpers ------------------------------------------------------------------------------
    def text(self, screen, s, pos, font=None, color=None) -> int:
        """Draw text; returns its width."""
        img = (font or self.font).render(s, True, color or self.th["text"])
        screen.blit(img, pos)
        return img.get_width()

    def wrap(self, s: str, width_px: int, font=None) -> list[str]:
        """Wrap text to a pixel width."""
        font = font or self.font
        words, lines, cur = s.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if font.size(t)[0] > width_px and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        return lines

    def panel(self, screen, rect, title) -> None:
        """A titled panel behind rect."""
        pygame.draw.rect(screen, self.th["panel"], pygame.Rect(rect.x - 3, rect.y - 24, rect.w + 6, rect.h + 28),
                         border_radius=6)
        self.text(screen, title, (rect.x + 2, rect.y - 19), self.head, self.th["accent"])

    # ---- the agent's mind -----------------------------------------------------------------------
    def _mind_layer(self, fd: dict) -> dict:
        """Map, fog band, belief glow, likely-area outlines and peak for this frame (cached per turn)."""
        m = fd["mind"]
        key = (fd["t"], tuple(m["pose"]), self.layers["belief"], self.layers["fog"])
        if self._mind and self._mind["key"] == key:
            return self._mind
        th = self.th
        wx0, _, wy0, _ = m["window"]
        occ_full, bel_full = m["occ"], np.asarray(m["belief"], dtype=np.float32)
        # zoom on what matters: explored cells, belief mass and the agent, inside the feasible window
        interest = ((occ_full == 1) | (occ_full == 2)) | (bel_full > bel_full.max() * 0.02)
        idx = np.argwhere(interest)
        px, py = m["pose"][0] - wx0, m["pose"][1] - wy0
        lo = np.minimum(idx.min(axis=0), [px, py]) - 6 if len(idx) else np.array([px - 20, py - 20])
        hi = np.maximum(idx.max(axis=0), [px, py]) + 6 if len(idx) else np.array([px + 20, py + 20])
        side = int(max(40, (hi - lo).max() + 1))
        c = (lo + hi) // 2
        lo = np.clip(c - side // 2, 0, None)
        hi = np.minimum(lo + side, occ_full.shape) - 1
        sl = (slice(int(lo[0]), int(hi[0]) + 1), slice(int(lo[1]), int(hi[1]) + 1))
        occ, belief = occ_full[sl], bel_full[sl]
        x0, y0 = wx0 + int(lo[0]), wy0 + int(lo[1])
        w, h = occ.shape
        rgb = np.zeros((w, h, 3), dtype=np.float32)
        fog = np.array(th["fog"]) if self.layers["fog"] else np.array(th["free"]) * 0.7
        for val, col in ((0, fog), (1, np.array(th["free"])), (2, np.array(th["wall"])), (3, np.array(th["oob"]))):
            rgb[occ == val] = col
        if self.layers["fog"]:  # one-cell darker band where the known map meets the unknown
            known = (occ == 1) | (occ == 2)
            kn = np.zeros_like(known)
            kn[1:, :] |= known[:-1, :]
            kn[:-1, :] |= known[1:, :]
            kn[:, 1:] |= known[:, :-1]
            kn[:, :-1] |= known[:, 1:]
            rgb[(occ == 0) & kn] = fog * PAL["fog_edge"]
        out = {"key": key, "x0": x0, "y0": y0, "size_cells": (w, h), "outlines": [], "peak": None}
        peak = float(bel_full.max())
        if self.layers["belief"] and peak > 0:
            floor = PAL["belief_floor"]
            a = np.clip((np.log10(np.maximum(belief / peak, 1e-12)) - math.log10(floor)) / -math.log10(floor), 0, 1)
            col = np.array(PAL["belief"]) + (np.array(PAL["belief_peak"]) - np.array(PAL["belief"])) * (a ** 3)[..., None]
            alpha = (a * PAL["belief_alpha"])[..., None]
            rgb = rgb * (1 - alpha) + col * alpha
            for mass, dashed, colour, width in ((0.9, True, PAL["area90"], 1), (0.5, False, PAL["area50"], 2)):
                mask = bel_full[sl] >= _likely_threshold(bel_full, mass)
                out["outlines"].append((_edges(mask), dashed, colour, width))
            pk = np.unravel_index(int(np.argmax(bel_full)), bel_full.shape)
            if sl[0].start <= pk[0] < sl[0].stop and sl[1].start <= pk[1] < sl[1].stop:
                out["peak"] = ((wx0 + int(pk[0]), wy0 + int(pk[1])), peak / max(float(bel_full.sum()), 1e-12))
        surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
        s = min(MIND.w / w, MIND.h / h)
        size = (max(1, int(w * s)), max(1, int(h * s)))
        out.update(surf=pygame.transform.scale(surf, size), s=s, size=size,
                   ox=MIND.x + (MIND.w - size[0]) // 2, oy=MIND.y + (MIND.h - size[1]) // 2)
        self._mind = out
        return out

    def _ping_frac(self, key) -> float:
        """How far the current scan's ring has travelled: 0..1 over PING_SECONDS of real time (1 when static)."""
        if self.static_ping:
            return 1.0
        now = pygame.time.get_ticks()
        if not self.ping or self.ping["key"] != key:
            self.ping = {"key": key, "start": now}
        return min(1.0, (now - self.ping["start"]) / (1000 * PING_SECONDS))

    def draw_mind(self, screen, fd: dict, tick: int) -> None:
        """Draw the agent's view (belief, map, plan, scans)."""
        m = fd["mind"]
        th = self.th
        self.panel(screen, MIND, "Agent's mind  (its own frame; fog = unknown)")
        L = self._mind_layer(fd)
        ox, oy, s, size, x0, y0 = L["ox"], L["oy"], L["s"], L["size"], L["x0"], L["y0"]
        screen.blit(L["surf"], (ox, oy))

        def P(c):
            return (ox + (c[0] - x0 + 0.5) * s, oy + (c[1] - y0 + 0.5) * s)

        clip = screen.get_clip()
        screen.set_clip(pygame.Rect(ox, oy, *size))
        for segs, dashed, colour, width in L["outlines"]:
            for ax, ay, bx, by in segs:
                p0 = (ox + ax * s, oy + ay * s)
                p1 = (ox + bx * s, oy + by * s)
                if dashed:
                    p0, p1 = ((p0[0] * 0.75 + p1[0] * 0.25), (p0[1] * 0.75 + p1[1] * 0.25)), \
                             ((p0[0] * 0.25 + p1[0] * 0.75), (p0[1] * 0.25 + p1[1] * 0.75))
                pygame.draw.line(screen, colour, p0, p1, width)
        for hx, hy in m.get("hearths", []):
            spr = scaled(self.sprites.get("hearth", tick // 6), max(8, int(s * 2.2)), self.cache)
            cx, cy = P((hx, hy))
            screen.blit(spr, (cx - spr.get_width() / 2, cy - spr.get_height() / 2))
        if self.layers["path"] and m.get("path"):
            pts = [P(m["pose"])] + [P(c) for c in m["path"]]
            if len(pts) > 1:
                pygame.draw.lines(screen, th["accent"], False, pts, 2)
        if m.get("goal"):
            cx, cy = P(m["goal"])
            r = max(5, s * 2)
            pygame.draw.circle(screen, PAL["goal"], (cx, cy), r, 1)
            pygame.draw.line(screen, PAL["goal"], (cx - r - 3, cy), (cx + r + 3, cy))
            pygame.draw.line(screen, PAL["goal"], (cx, cy - r - 3), (cx, cy + r + 3))
        for d in m.get("fused", []):
            cx, cy = P(d["pos"])
            col = PAL["guess"].get(d.get("guess", "unknown"), PAL["guess"]["unknown"])
            pygame.draw.circle(screen, col, (cx, cy), max(3, d.get("sigma", 1) * s), 2)
        self._draw_scan(screen, fd, P, s)
        if L["peak"]:
            (pc, pv) = L["peak"]
            cx, cy = P(pc)
            r = max(4, s * 0.9)
            pygame.draw.polygon(screen, PAL["outline"], [(cx, cy - r - 1), (cx + r + 1, cy), (cx, cy + r + 1), (cx - r - 1, cy)])
            pygame.draw.polygon(screen, PAL["peak"], [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)])
            self.text(screen, f"P = {pv:.2f}" if pv >= 0.01 else f"P = {pv:.3f}", (cx + r + 4, cy - 7), self.small, PAL["peak"])
        spr = scaled(self.sprites.get(th["rescuer_sprite"], tick // 8), max(10, int(s * 3)), self.cache)
        if m["facing"] == "W":
            spr = pygame.transform.flip(spr, True, False)
        cx, cy = P(m["pose"])
        screen.blit(spr, (cx - spr.get_width() / 2, cy - spr.get_height() + spr.get_width() / 2))
        screen.set_clip(clip)
        fx0, fx1, fy0, fy1 = m["window"]
        w, h = L["size_cells"]
        self.text(screen, f"view x{x0}-{x0 + w - 1} y{y0}-{y0 + h - 1}   feasible window x{fx0}-{fx1} y{fy0}-{fy1}",
                  (MIND.x + 4, MIND.bottom - 16), self.small, th["dim"])
        if self.layers["legend"]:
            self._legend(screen)

    def _draw_scan(self, screen, fd: dict, P, s: float) -> None:
        """The sonar ping of this turn's scan and what it found (revealed as the ring passes)."""
        m = fd["mind"]
        scan = m.get("scan")
        if not scan:
            return
        kind = scan["kind"]
        col = PAL["sensor"].get(kind, PAL["goal"])
        if kind == "cv":
            dx, dy = DIRS[scan["facing"]]
            lx, ly = -dy, dx
            px, py = scan["pose"]
            for c in [(px + dx, py + dy)] + [(px + 2 * dx + k * lx, py + 2 * dy + k * ly) for k in (-1, 0, 1)]:
                cx, cy = P(c)
                pygame.draw.rect(screen, col, (cx - s / 2, cy - s / 2, max(2, s), max(2, s)), 2)
            return
        if kind not in self.ranges:
            return
        frac = self._ping_frac((fd["t"], kind, tuple(scan["pose"])))
        R = self.ranges[kind] * s
        reach = R * frac
        origin = P(scan["pose"])
        fx = self._fx
        fx.fill((0, 0, 0, 0))
        local = (origin[0] - MIND.x, origin[1] - MIND.y)
        if self.static_ping:
            pygame.draw.circle(fx, col + (PAL["ping_static_alpha"],), local, R, 2)
        elif frac < 1:
            pygame.draw.circle(fx, col + (int(PAL["ping_alpha"] * (1 - frac) + 20),), local, max(2, reach), 2)
        readings = m.get("readings") or {}
        now = pygame.time.get_ticks()
        for b in readings.get("thermal", []) if kind == "thermal" else []:
            c = P(b["pos"])
            if math.dist(c, origin) <= reach + 1:
                pygame.draw.circle(fx, col + (200,), (c[0] - MIND.x, c[1] - MIND.y), min(7, max(2, b.get("radius", 1) * s * 0.22)))
        for b in readings.get("radar", []) if kind == "radar" else []:
            c = P(b["pos"])
            if math.dist(c, origin) <= reach + 1:
                ph = 0.5 if self.static_ping else (now % 700) / 700
                lc = (c[0] - MIND.x, c[1] - MIND.y)
                pygame.draw.circle(fx, col + (230,), lc, max(2, s * 0.5))
                pygame.draw.circle(fx, col + (int(200 * (1 - ph)),), lc, max(3, s * (0.8 + 2 * ph)), 1)
        for b in (readings.get("lidar", {}) or {}).get("blobs", []) if kind == "lidar" else []:
            c = P(b["pos"])
            if math.dist(c, origin) <= reach + 1 and b["shape"] != "box":
                pygame.draw.rect(fx, col + (230,), (c[0] - MIND.x - s / 2, c[1] - MIND.y - s / 2, max(3, s), max(3, s)), 1)
        screen.blit(fx, MIND.topleft)

    def _legend(self, screen) -> None:
        rows = [("glow", "belief: brighter = more likely"), ("solid", "50% likely area"), ("dashed", "90% likely area"),
                ("diamond", "belief peak, its probability")] + \
               [("ring:" + k, f"{k} scan") for k in ("thermal", "radar", "lidar", "cv")] + \
               [("guess:" + g, f"track: {g}") for g in ("target?", "living?", "dog", "fire?", "hearth", "armour", "unknown")]
        box = pygame.Rect(MIND.x + 8, MIND.y + 8, 230, 14 * len(rows) + 12)
        bg = pygame.Surface(box.size, pygame.SRCALPHA)
        bg.fill(PAL["legend_bg"])
        screen.blit(bg, box.topleft)
        for i, (kind, label) in enumerate(rows):
            y = box.y + 8 + i * 14
            cx = box.x + 14
            if kind == "glow":
                for j in range(5):
                    a = j / 4
                    c = [int(PAL["belief"][k] + (PAL["belief_peak"][k] - PAL["belief"][k]) * a ** 3) for k in range(3)]
                    pygame.draw.rect(screen, c, (box.x + 6 + j * 4, y + 2, 4, 9))
            elif kind in ("solid", "dashed"):
                if kind == "solid":
                    pygame.draw.line(screen, PAL["area50"], (box.x + 6, y + 6), (box.x + 24, y + 6), 2)
                else:
                    for j in range(3):
                        pygame.draw.line(screen, PAL["area90"], (box.x + 6 + 7 * j, y + 6), (box.x + 10 + 7 * j, y + 6), 1)
            elif kind == "diamond":
                pygame.draw.polygon(screen, PAL["peak"], [(cx, y + 1), (cx + 5, y + 6), (cx, y + 11), (cx - 5, y + 6)])
            elif kind.startswith("ring:"):
                pygame.draw.circle(screen, PAL["sensor"][kind[5:]], (cx, y + 6), 5, 2)
            else:
                pygame.draw.circle(screen, PAL["guess"][kind[6:]], (cx, y + 6), 5, 2)
            self.text(screen, label, (box.x + 32, y), self.small, self.th["text"])

    # ---- ground truth ---------------------------------------------------------------------------
    def _entities(self, fd):
        tr = fd["truth"]
        dogs = {d[0]: (d[1], d[2]) for d in tr["dogs"]}
        out = []
        for e in self.static["entities"]:
            if e["kind"] == "target":
                continue
            pos = dogs.get(e["id"], tuple(e["pos"])) if e["kind"] == "dog" else tuple(e["pos"])
            out.append((e["kind"], pos, e["id"]))
        return out

    def _light(self, screen, centre, cell: float, colour, flicker: float) -> None:
        """A small warm light: 4 concentric circles of falling opacity, radius about 1.5 cells."""
        r = 1.5 * cell
        size = int(2 * r + 4)
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        for k, a in enumerate((16, 24, 34, 46)):
            pygame.draw.circle(glow, colour + (int(a * flicker),), (size / 2, size / 2), r * (1 - 0.22 * k))
        screen.blit(glow, (centre[0] - size / 2, centre[1] - size / 2))

    def draw_truth(self, screen, fd: dict, tick: int, hidden: bool = False) -> None:
        """Draw the true world."""
        th = self.th
        self.panel(screen, TRUTH, "Ground truth" + ("  (full map, M)" if self.fullmap else "  (camera on the rescuer, M = full map)"))
        if hidden or not self.layers["truth"]:
            pygame.draw.rect(screen, th["fog"], TRUTH)
            msg = "hidden until the mission ends" if hidden else "truth layer off (T)"
            self.text(screen, msg, (TRUTH.x + 140, TRUTH.centery - 8), self.font, th["dim"])
            return
        tr = fd["truth"]
        rx, ry = tr["rescuer"]
        if self.fullmap:
            cell = TRUTH.w / self.n
            screen.blit(self._castle_small, TRUTH.topleft)
            cx0, cy0 = 0, 0
        else:
            cell = 16
            cx0 = min(max(rx - CAM // 2, 0), self.n - CAM)
            cy0 = min(max(ry - CAM // 2, 0), self.n - CAM)
            screen.blit(self._castle, TRUTH.topleft, pygame.Rect(cx0 * 16, cy0 * 16, CAM * 16, CAM * 16))

        def P(c):
            return (TRUTH.x + (c[0] - cx0) * cell, TRUTH.y + (c[1] - cy0) * cell)

        clip = screen.get_clip()
        screen.set_clip(TRUTH)
        trail = fd.get("trail") or []
        if len(trail) > 1:
            pygame.draw.lines(screen, PAL["trail"], False,
                              [(P(c)[0] + cell / 2, P(c)[1] + cell / 2) for c in trail[-600:]], 1 if self.fullmap else 2)
        if self.fullmap:  # sprites would be 5 px: draw clear markers instead
            for kind, pos, eid in self._entities(fd):
                x, y = P(pos)
                pygame.draw.circle(screen, PAL["entity_dot"][kind], (x + cell / 2, y + cell / 2), 3)
            for pos, col in ((tr["target"], PAL["target"] if tr["alive"] else PAL["dead"]), (tr["rescuer"], PAL["rescuer"])):
                x, y = P(pos)
                pygame.draw.circle(screen, PAL["outline"], (x + cell / 2, y + cell / 2), 6)
                pygame.draw.circle(screen, col, (x + cell / 2, y + cell / 2), 5)
            screen.set_clip(clip)
            return
        size = max(5, int(cell))
        ents = self._entities(fd)
        for kind, pos, eid in ents:  # lights first, so sprites sit on top of them
            if kind in ("torch", "hearth"):
                x, y = P(pos)
                flicker = 0.85 + 0.15 * math.sin(tick * 0.35 + eid * 1.7) if kind == "torch" else 0.95
                self._light(screen, (x + cell / 2, y + cell / 2), cell, PAL["torch_light" if kind == "torch" else "hearth_light"], flicker)
        for kind, pos, eid in ents:
            spr = scaled(self.sprites.get(kind, (tick + eid * 3) // 5), size, self.cache)
            x, y = P(pos)
            screen.blit(spr, (x, y + cell - spr.get_height()))
        tx, ty = P(tr["target"])
        tspr = self.sprites.get(th["target_sprite"], tick // 8)
        if not tr["alive"]:
            tspr = pygame.transform.rotate(tspr, 90)
        tspr = scaled(tspr, size, self.cache)
        screen.blit(tspr, (tx, ty + cell - tspr.get_height()))
        bw = max(12, size)
        pygame.draw.rect(screen, PAL["energy_bg"], (tx + cell / 2 - bw / 2, ty - 6, bw, 4))
        e = max(0.0, min(1.0, tr["energy"]))
        ec = [int(PAL["energy_lo"][k] + (PAL["energy_hi"][k] - PAL["energy_lo"][k]) * e) for k in range(3)]
        pygame.draw.rect(screen, ec, (tx + cell / 2 - bw / 2, ty - 6, bw * e, 4))
        rspr = scaled(self.sprites.get(th["rescuer_sprite"], tick // 8), size, self.cache)
        if tr["facing"] == "W":
            rspr = pygame.transform.flip(rspr, True, False)
        x, y = P((rx, ry))
        screen.blit(rspr, (x, y + cell - rspr.get_height()))
        screen.set_clip(clip)
        self._minimap(screen, fd, (TRUTH.right - 108, TRUTH.y + 6), (cx0, cy0))

    def _minimap(self, screen, fd, pos, cam):
        mm = pygame.transform.scale(self._castle_small, (100, 100))
        mm.set_alpha(220)
        screen.blit(mm, pos)
        k = 100 / self.n
        tr = fd["truth"]
        pygame.draw.rect(screen, PAL["minimap_frame"], (pos[0] + cam[0] * k, pos[1] + cam[1] * k, CAM * k, CAM * k), 1)
        pygame.draw.circle(screen, PAL["target"], (pos[0] + tr["target"][0] * k, pos[1] + tr["target"][1] * k), 2)
        pygame.draw.circle(screen, PAL["rescuer"], (pos[0] + tr["rescuer"][0] * k, pos[1] + tr["rescuer"][1] * k), 2)

    # ---- receipts and status --------------------------------------------------------------------
    @staticmethod
    def _collapse(receipts: list[dict]) -> list[dict]:
        """Merge runs of plain moves into one row: MOVE x4 -0.8%."""
        out: list[dict] = []
        for r in receipts:
            if r["kind"] == "move" and not r["res"] and out and out[-1]["kind"] == "move" and not out[-1]["res"]:
                last = out[-1]
                last.update(t1=r["t"], n=last.get("n", 1) + 1, cost=last["cost"] + r["cost"], bal=r["bal"])
            else:
                out.append(dict(r))
        return out

    def draw_receipts(self, screen, receipts: list[dict]) -> None:
        """Draw the action receipts."""
        self.panel(screen, RECEIPT, "Receipts")
        pygame.draw.rect(screen, self.th["inset"], RECEIPT)
        line_h = 13
        rows = []
        for r in self._collapse(receipts[-120:]):
            if r.get("n", 1) > 1:
                head = f"t={r['t']}-{r['t1']} MOVE x{r['n']} -{r['cost']:.1f}%"
            else:
                cost = f"-{r['cost']:.1f}%" if r["cost"] else (r["res"] if r["kind"] == "charge" else "0%")
                head = f"t={r['t']} {r['act']} {cost}"
            body = r["res"] if r["res"] and r["kind"] != "charge" else ""
            lines = self.wrap(head + ("  " + body if body else ""), RECEIPT.w - 54, self.small)
            for j, l in enumerate(lines):
                rows.append((l, r, j == 0))
        rows = rows[-(RECEIPT.h // line_h):]
        for i, (l, r, first) in enumerate(rows):
            y = RECEIPT.y + 2 + i * line_h
            col = PAL["receipt"].get(r["kind"], self.th["dim"])
            if r["flag"]:  # findings stand out: a tinted row and a marker
                tint = pygame.Surface((RECEIPT.w, line_h), pygame.SRCALPHA)
                tint.fill(col + (40,))
                screen.blit(tint, (RECEIPT.x, y - 1))
                if first:
                    pygame.draw.rect(screen, col, (RECEIPT.x + 2, y + 3, 4, 6))
            self.text(screen, l, (RECEIPT.x + 9, y), self.small, col)
            if first:
                b = f"{r['bal']:.1f}%"
                self.text(screen, b, (RECEIPT.right - 4 - self.small.size(b)[0], y), self.small, self.th["dim"])

    def _bar(self, screen, rect, frac, colour) -> None:
        pygame.draw.rect(screen, PAL["bar_bg"], rect, border_radius=3)
        if frac > 0:
            pygame.draw.rect(screen, colour, (rect.x, rect.y, rect.w * min(frac, 1.0), rect.h), border_radius=3)

    def _life_bar(self, screen, x, y, fd: dict, hidden: bool) -> None:
        life = fd["mind"].get("life") or {}
        tr = fd["truth"]
        true_left = None if hidden else float(tr.get("remaining_life", 0) or 0)
        mean, p10, p90 = (float(life.get(k, 0) or 0) for k in ("mean", "p10", "p90"))
        top = max(p90, mean, true_left or 0, 1.0) * 1.08
        rect = pygame.Rect(x, y + 14, 300, 16)
        pygame.draw.rect(screen, PAL["bar_bg"], rect, border_radius=3)
        pygame.draw.rect(screen, PAL["life_band"], (x + 300 * p10 / top, y + 14, 300 * (p90 - p10) / top, 16), border_radius=3)
        mx = x + 300 * mean / top
        pygame.draw.line(screen, PAL["life"], (mx, y + 13), (mx, y + 30), 3)
        label = f"LIFE LEFT  est {mean:.0f}  (10-90%: {p10:.0f}-{p90:.0f})"
        if true_left is not None:
            tx = x + 300 * true_left / top
            pygame.draw.polygon(screen, PAL["life_true"], [(tx - 5, y + 37), (tx + 5, y + 37), (tx, y + 31)])
            pygame.draw.line(screen, PAL["life_true"], (tx, y + 14), (tx, y + 30), 1)
            label += f"  true {true_left:.0f}"
        self.text(screen, label, (x, y), self.small, self.th["dim"])

    def _timeline(self, screen, t: int) -> None:
        n = max(self.n_turns, t + 1, 2)
        r = TIMELINE
        pygame.draw.rect(screen, PAL["timeline_bg"], r, border_radius=3)
        order = ("charge", "thermal", "radar", "lidar", "cv", "override")
        for tt, kinds in self.events.items():
            x = r.x + r.w * tt / (n - 1)
            for k in sorted(set(kinds), key=order.index):
                top = r.y if k in ("cv", "override") else r.y + 3
                pygame.draw.line(screen, PAL["event"][k], (x, top), (x, r.bottom - 1), 2 if k in ("cv", "override") else 1)
        cx = r.x + r.w * t / (n - 1)
        pygame.draw.line(screen, PAL["cursor"], (cx, r.y - 4), (cx, r.bottom + 3), 2)
        pygame.draw.polygon(screen, PAL["cursor"], [(cx - 4, r.y - 6), (cx + 4, r.y - 6), (cx, r.y - 1)])
        legend = [("detections (thermal, radar, LIDAR)", ("thermal", "radar", "lidar")), ("camera label", ("cv",)),
                  ("charge", ("charge",)), ("override", ("override",))]
        lx = r.x
        for name, kinds in legend:
            for k in kinds:
                pygame.draw.rect(screen, PAL["event"][k], (lx, r.y - 15, 8, 6))
                lx += 10
            lx += 3 + self.text(screen, name, (lx + 3, r.y - 19), self.small, self.th["dim"]) + 14

    def draw_bar(self, screen, fd: dict, hidden: bool = False, extra_help: str = "") -> None:
        """Draw the status bar: battery, life, mode, turn, override, reasoning, timeline, help."""
        th = self.th
        pygame.draw.rect(screen, th["panel"], BAR, border_radius=6)
        x, y = BAR.x + 12, BAR.y + 10
        bat = fd["battery"]
        self.text(screen, "BATTERY", (x, y), self.small, th["dim"])
        bc = [int(PAL["battery_lo"][k] + (PAL["battery_hi"][k] - PAL["battery_lo"][k]) * bat / 100) for k in range(3)]
        self._bar(screen, pygame.Rect(x, y + 14, 260, 16), bat / 100, bc)
        self.text(screen, f"{bat:.1f}%", (x + 266, y + 15))
        self._life_bar(screen, x + 350, y, fd, hidden)
        mode = fd.get("mode") or "-"
        pygame.draw.rect(screen, PAL["mode"].get(mode, PAL["mode_other"]), (x + 990, y + 10, 96, 22), border_radius=4)
        self.text(screen, mode.upper(), (x + 998, y + 14), self.font, PAL["mode_text"])
        self.text(screen, f"TURN {fd['t']}", (x + 1100, y + 9), self.num)
        tr, m = fd["truth"], fd["mind"]
        row2 = y + 42
        info = f"p_move {m.get('p_move', 0):.2f}"
        if not hidden:
            info += f"   truth: {th['target_name']} energy {tr['energy']:.2f}" + ("" if tr["alive"] else "  (DEAD)")
        self.text(screen, info, (x, row2 + 6), self.font, PAL["life_true"] if not hidden else th["dim"])
        ov = fd.get("override")
        if ov:  # its own row, right of the info line: never under the turn counter
            p = ov["plans"]
            plans = "  |  ".join(f"{k} {v:.2f}" for k, v in p.items())
            msg = f"ENDGAME OVERRIDE   {plans}   ->  {ov['choice'].upper()}"
            w = self.font.size(msg)[0] + 24
            pygame.draw.rect(screen, PAL["override_bg"], (BAR.right - w - 12, row2, w, 26), border_radius=6)
            self.text(screen, msg, (BAR.right - w, row2 + 6), self.font, PAL["override_text"])
        lines = self.wrap("Reasoning: " + (fd.get("reason") or ""), BAR.w - 30)
        for i, line in enumerate(lines[:2]):
            self.text(screen, line, (x, y + 76 + 17 * i), self.font, th["accent"] if i == 0 else th["text"])
        self._timeline(screen, fd["t"])
        help_ = "space play/pause  <-/-> step  +/- speed  click timeline: jump  B belief  F fog  P path  T truth  " \
                "M full map  L legend  Q quit"
        self.text(screen, extra_help or help_, (x, BAR.bottom - 18), self.small, th["dim"])

    def draw(self, screen, fd: dict, receipts: list[dict], tick: int, hidden: bool = False, extra_help: str = "") -> None:
        """Draw a full frame."""
        th = self.th
        screen.fill(th["bg"])
        title = th["title"]
        self.text(screen, title, (12, 10), self.title, th["accent"])
        sub = f"{self.cfg.get('world', {}).get('weather', {}).get('resolved', {}).get('location', '')} weather " \
              f"{fd.get('weather', '')}  seed {self.cfg.get('seed')}  art: {self.sprites.pack}"
        self.text(screen, sub, (14 + self.title.size(title)[0] + 16, 16), self.small, th["dim"])
        self.draw_mind(screen, fd, tick)
        self.draw_truth(screen, fd, tick, hidden)
        self.draw_receipts(screen, receipts)
        self.draw_bar(screen, fd, hidden, extra_help)
        if fd.get("outcome"):
            word = {"rescued": "RESCUED!", "target_died": "TOO LATE...", "battery_depleted": "OUT OF BATTERY",
                    "step_cap": "TIME UP"}.get(fd["outcome"], fd["outcome"].upper())
            col = PAL["outcome_good"] if fd["outcome"] == "rescued" else PAL["outcome_bad"]
            img = self.huge.render(word, True, col)
            box = img.get_rect(center=(TRUTH.centerx, TRUTH.centery))
            pygame.draw.rect(screen, PAL["outcome_box"], box.inflate(30, 16), border_radius=8)
            screen.blit(img, box)


def handle_common_keys(viewer: Viewer, key) -> bool:
    """Layer toggles shared by replay and play. Returns True if the key was used."""
    toggles = {pygame.K_b: "belief", pygame.K_f: "fog", pygame.K_p: "path", pygame.K_t: "truth", pygame.K_l: "legend"}
    if key in toggles:
        viewer.layers[toggles[key]] = not viewer.layers[toggles[key]]
        return True
    if key == pygame.K_m:
        viewer.fullmap = not viewer.fullmap
        return True
    return False


def weather_str(w: dict) -> str:
    """Short weather description."""
    return f"{w.get('temperature', '?')}C {w.get('humidity', '?')}%RH rain {w.get('rain', 0)}mm"
