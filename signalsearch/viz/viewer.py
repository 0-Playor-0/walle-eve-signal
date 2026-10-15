"""Pygame viewer. Draws one FrameData dict (built from logs by replay.py or live by play.py):

  left   - the agent's mind: occupancy (unknown as fog), belief heatmap, known hearths, planned path,
           a flash of each scan's coverage, fused detections, current goal
  right  - ground truth: the castle with sprites, rescuer, target with energy bar, dogs, braziers...
  bottom - battery, turn / life (true and estimated), reasoning sentence, override banner
  far right - receipts: every battery "purchase"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from ..common.actions import DIRS
from .assets import Sprites, scaled
from .themes import theme

W, H = 1280, 760
MIND = pygame.Rect(10, 64, 500, 500)
TRUTH = pygame.Rect(520, 64, 500, 500)
RECEIPT = pygame.Rect(1030, 64, 240, 500)
BAR = pygame.Rect(10, 576, 1260, 176)
CAM = 31  # truth camera: cells per side
ASSETS = Path(__file__).resolve().parent.parent.parent / "assets"

GUESS_COL = {"target?": (255, 90, 200), "target": (255, 60, 180), "dog?": (190, 130, 70), "dog": (190, 130, 70),
             "fire?": (255, 150, 40), "torch": (255, 150, 40), "hearth?": (255, 90, 40), "hearth": (255, 90, 40),
             "armour": (150, 200, 230), "humanoid (armour or target)?": (120, 220, 255), "living?": (255, 120, 140),
             "warm source?": (255, 200, 120), "unknown": (220, 220, 220)}
MODE_COL = {"search": (90, 170, 110), "approach": (90, 160, 230), "return": (220, 170, 60),
            "dash": (230, 80, 60), "lean": (230, 110, 60)}


def receipt(obs: dict) -> str:
    a = obs["action"]
    r = obs.get("readings", {})
    if a == "spawn":
        return f"t=0 SPAWN on hearth | bal {obs['battery']:.1f}%"
    if "lidar" in r:
        res = f"{len(r['lidar']['free'])} free, {len(r['lidar']['walls'])} walls, {len(r['lidar']['blobs'])} shapes"
    elif "radar" in r:
        res = f"{len(r['radar'])} heartbeat{'s' if len(r['radar']) != 1 else ''}"
    elif "thermal" in r:
        res = f"{len(r['thermal'])} warm blob{'s' if len(r['thermal']) != 1 else ''}"
    elif "cv" in r:
        labs = [x["label"] for x in r["cv"]["labels"]]
        res = ", ".join(labs) if labs else "nothing"
    elif a == "charge":
        res = f"+{obs.get('charged', 0):.1f}%"
    elif obs.get("blocked"):
        res = "blocked!"
    else:
        res = ""
    cost = f"-{obs['cost']:.1f}%" if obs["cost"] else ("+" + f"{obs.get('charged', 0):.0f}%" if a == "charge" else "0%")
    return f"t={obs['t']} {a.upper()} {cost}{' -> ' + res if res else ''} | bal {obs['battery']:.1f}%"


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
        self.layers = {"belief": True, "fog": True, "path": True, "truth": True}
        self.fullmap = False
        pygame.font.init()
        self.font = pygame.font.SysFont("menlo,dejavusansmono,couriernew,monospace", 13)
        self.small = pygame.font.SysFont("menlo,dejavusansmono,couriernew,monospace", 11)
        self.big = pygame.font.SysFont("avenirnext,helveticaneue,arial", 22, bold=True)
        self.huge = pygame.font.SysFont("avenirnext,helveticaneue,arial", 40, bold=True)
        sens = cfg["agent_assumed"]["sensors"]
        self.ranges = {"lidar": sens["lidar"]["range"], "radar": sens["radar"]["range"], "thermal": sens["thermal"]["range"]}
        self._castle = self._render_castle()
        self._castle_small = pygame.transform.smoothscale(self._castle, (TRUTH.w, TRUTH.h))

    # ---- static castle ------------------------------------------------------------------------
    def _render_castle(self) -> pygame.Surface:
        n, T = self.n, 16
        surf = pygame.Surface((n * T, n * T))
        dim = pygame.Surface((T, T))
        dim.fill((120, 120, 128))
        for x in range(n):
            for y in range(n):
                if self.walls[x, y]:
                    surf.blit(scaled(self.sprites.get("wall"), T, self.cache), (x * T, y * T))
                else:
                    surf.blit(scaled(self.sprites.variant("floor", x, y), T, self.cache), (x * T, y * T))
                    surf.blit(dim, (x * T, y * T), special_flags=pygame.BLEND_MULT)
        shadow = pygame.Surface((T, 4), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 110))
        for x, y in np.argwhere(self.walls[:, :-1] & ~self.walls[:, 1:]):  # wall face casts a shadow below
            surf.blit(shadow, (x * T, (y + 1) * T))
        if self.th["name"] == "walle":  # rusty tint for the junkyard skin
            tint = pygame.Surface(surf.get_size())
            tint.fill(self.th["tint_mult"])
            surf.blit(tint, (0, 0), special_flags=pygame.BLEND_MULT)
        return surf

    # ---- helpers ------------------------------------------------------------------------------
    def text(self, screen, s, pos, font=None, color=None):
        img = (font or self.font).render(s, True, color or self.th["text"])
        screen.blit(img, pos)
        return img.get_width()

    def wrap(self, s: str, width_px: int, font=None) -> list[str]:
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

    def panel(self, screen, rect, title):
        pygame.draw.rect(screen, self.th["panel"], pygame.Rect(rect.x - 3, rect.y - 24, rect.w + 6, rect.h + 28),
                         border_radius=6)
        self.text(screen, title, (rect.x + 2, rect.y - 20), self.font, self.th["accent"])

    # ---- the agent's mind -----------------------------------------------------------------------
    def draw_mind(self, screen, fd: dict, tick: int):
        m = fd["mind"]
        self.panel(screen, MIND, "AGENT'S MIND  (its own frame; fog = unknown)")
        wx0, _, wy0, _ = m["window"]
        occ_full, bel_full = m["occ"], m["belief"].astype(np.float32)
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
        occ = occ_full[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
        belief = bel_full[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
        x0, y0 = wx0 + int(lo[0]), wy0 + int(lo[1])
        x1, y1 = x0 + occ.shape[0] - 1, y0 + occ.shape[1] - 1
        w, h = occ.shape
        th = self.th
        rgb = np.zeros((w, h, 3), dtype=np.float32)
        fog = np.array(th["fog"]) if self.layers["fog"] else np.array(th["free"]) * 0.7
        for val, col in ((0, fog), (1, np.array(th["free"])), (2, np.array(th["wall"])), (3, np.array(th["oob"]))):
            rgb[occ == val] = col
        if self.layers["belief"]:
            b = belief
            v = np.sqrt(np.clip(b / max(float(b.max()), 1e-12), 0, 1))
            heat = np.stack([255 * np.clip(v * 1.6, 0, 1), 255 * np.clip(v * 1.6 - 0.55, 0, 1),
                             255 * np.clip(v * 2.2 - 1.6, 0, 1)], axis=-1)
            a = np.clip(v * 1.1, 0, 0.92)[..., None]
            rgb = rgb * (1 - a) + heat * a
        surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
        s = min(MIND.w / w, MIND.h / h)
        size = (max(1, int(w * s)), max(1, int(h * s)))
        ox = MIND.x + (MIND.w - size[0]) // 2
        oy = MIND.y + (MIND.h - size[1]) // 2
        screen.blit(pygame.transform.scale(surf, size), (ox, oy))

        def P(c):
            return (ox + (c[0] - x0 + 0.5) * s, oy + (c[1] - y0 + 0.5) * s)

        clip = screen.get_clip()
        screen.set_clip(pygame.Rect(ox, oy, *size))
        scan = m.get("scan")
        if scan and scan["kind"] in self.ranges:
            r = self.ranges[scan["kind"]] * s
            ov = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            col = {"thermal": (255, 120, 40), "radar": (120, 220, 120), "lidar": (120, 180, 255)}[scan["kind"]]
            pygame.draw.circle(ov, col + (40,), P(scan["pose"]), r)
            pygame.draw.circle(ov, col + (160,), P(scan["pose"]), r, 2)
            screen.blit(ov, (0, 0))
        if scan and scan["kind"] == "cv":
            dx, dy = DIRS[scan["facing"]]
            lx, ly = -dy, dx
            px, py = scan["pose"]
            cells = [(px + dx, py + dy)] + [(px + 2 * dx + k * lx, py + 2 * dy + k * ly) for k in (-1, 0, 1)]
            for c in cells:
                cx, cy = P(c)
                pygame.draw.rect(screen, (255, 255, 120), (cx - s / 2, cy - s / 2, max(2, s), max(2, s)), 1)
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
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, 1)
            pygame.draw.line(screen, (255, 255, 255), (cx - r - 3, cy), (cx + r + 3, cy))
            pygame.draw.line(screen, (255, 255, 255), (cx, cy - r - 3), (cx, cy + r + 3))
        for d in m.get("fused", []):
            cx, cy = P(d["pos"])
            col = GUESS_COL.get(d.get("guess", "unknown"), (220, 220, 220))
            rr = max(3, d.get("sigma", 1) * s)
            pygame.draw.circle(screen, col, (cx, cy), rr, 2)
        spr = self.sprites.get(th["rescuer_sprite"], tick // 8)
        spr = scaled(spr, max(10, int(s * 3)), self.cache)
        if m["facing"] == "W":
            spr = pygame.transform.flip(spr, True, False)
        cx, cy = P(m["pose"])
        screen.blit(spr, (cx - spr.get_width() / 2, cy - spr.get_height() + spr.get_width() / 2))
        screen.set_clip(clip)
        fx0, fx1, fy0, fy1 = m["window"]
        self.text(screen, f"view x{x0}-{x1} y{y0}-{y1}   feasible window x{fx0}-{fx1} y{fy0}-{fy1}",
                  (MIND.x + 4, MIND.bottom - 16), self.small, th["dim"])

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

    def draw_truth(self, screen, fd: dict, tick: int, hidden: bool = False):
        th = self.th
        self.panel(screen, TRUTH, "GROUND TRUTH" + ("  (full map, M)" if self.fullmap else "  (camera on the rescuer, M = full map)"))
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
            pygame.draw.lines(screen, (90, 200, 255), False,
                              [(P(c)[0] + cell / 2, P(c)[1] + cell / 2) for c in trail[-600:]], 1 if self.fullmap else 2)
        if self.fullmap:  # sprites would be 5 px: draw clear markers instead
            dot = {"dog": (190, 130, 70), "torch": (255, 150, 40), "hearth": (255, 70, 40), "armour": (160, 200, 230)}
            for kind, pos, eid in self._entities(fd):
                x, y = P(pos)
                pygame.draw.circle(screen, dot[kind], (x + cell / 2, y + cell / 2), 3)
            for pos, col in ((tr["target"], (255, 80, 200) if tr["alive"] else (120, 120, 120)), (tr["rescuer"], (90, 200, 255))):
                x, y = P(pos)
                pygame.draw.circle(screen, (0, 0, 0), (x + cell / 2, y + cell / 2), 6)
                pygame.draw.circle(screen, col, (x + cell / 2, y + cell / 2), 5)
            screen.set_clip(clip)
            return
        size = max(5, int(cell))
        for kind, pos, eid in self._entities(fd):
            spr = scaled(self.sprites.get(kind, (tick + eid * 3) // 5), size, self.cache)
            x, y = P(pos)
            if kind == "torch" and not self.fullmap:  # flickering glow
                glow = pygame.Surface((48, 48), pygame.SRCALPHA)
                pygame.draw.circle(glow, (255, 160, 60, 40 + 20 * ((tick // 3) % 3)), (24, 24), 22)
                screen.blit(glow, (x + cell / 2 - 24, y + cell / 2 - 24))
            screen.blit(spr, (x, y + cell - spr.get_height()))
        tx, ty = P(tr["target"])
        tspr = self.sprites.get(th["target_sprite"], tick // 8)
        if not tr["alive"]:
            tspr = pygame.transform.rotate(tspr, 90)
        tspr = scaled(tspr, size, self.cache)
        screen.blit(tspr, (tx, ty + cell - tspr.get_height()))
        bw = max(12, size)
        pygame.draw.rect(screen, (40, 0, 0), (tx + cell / 2 - bw / 2, ty - 6, bw, 4))
        e = max(0.0, min(1.0, tr["energy"]))
        pygame.draw.rect(screen, (int(255 * (1 - e)), int(220 * e), 60), (tx + cell / 2 - bw / 2, ty - 6, bw * e, 4))
        rspr = scaled(self.sprites.get(th["rescuer_sprite"], tick // 8), size, self.cache)
        if tr["facing"] == "W":
            rspr = pygame.transform.flip(rspr, True, False)
        x, y = P((rx, ry))
        screen.blit(rspr, (x, y + cell - rspr.get_height()))
        screen.set_clip(clip)
        if not self.fullmap:
            self._minimap(screen, fd, (TRUTH.right - 108, TRUTH.y + 6), (cx0, cy0))

    def _minimap(self, screen, fd, pos, cam):
        mm = pygame.transform.scale(self._castle_small, (100, 100))
        mm.set_alpha(220)
        screen.blit(mm, pos)
        k = 100 / self.n
        tr = fd["truth"]
        pygame.draw.rect(screen, (255, 255, 255), (pos[0] + cam[0] * k, pos[1] + cam[1] * k, CAM * k, CAM * k), 1)
        pygame.draw.circle(screen, (255, 80, 200), (pos[0] + tr["target"][0] * k, pos[1] + tr["target"][1] * k), 2)
        pygame.draw.circle(screen, (90, 200, 255), (pos[0] + tr["rescuer"][0] * k, pos[1] + tr["rescuer"][1] * k), 2)

    # ---- receipts and status --------------------------------------------------------------------
    def draw_receipts(self, screen, receipts: list[str]):
        self.panel(screen, RECEIPT, "RECEIPTS")
        pygame.draw.rect(screen, (24, 22, 28), RECEIPT)
        line_h = 13
        lines = []
        for r in receipts[-60:]:
            lines += self.wrap(r, RECEIPT.w - 8, self.small)
        lines = lines[-(RECEIPT.h // line_h):]
        for i, l in enumerate(lines):
            col = self.th["text"] if i == len(lines) - 1 else self.th["dim"]
            self.text(screen, l, (RECEIPT.x + 4, RECEIPT.y + 2 + i * line_h), self.small, col)

    def draw_bar(self, screen, fd: dict, hidden: bool = False, extra_help: str = ""):
        th = self.th
        pygame.draw.rect(screen, th["panel"], BAR, border_radius=6)
        x, y = BAR.x + 12, BAR.y + 10
        bat = fd["battery"]
        self.text(screen, "BATTERY", (x, y), self.small, th["dim"])
        pygame.draw.rect(screen, (50, 50, 56), (x, y + 14, 300, 16), border_radius=3)
        c = (int(230 - 2 * bat), int(60 + 1.6 * bat), 70)
        pygame.draw.rect(screen, c, (x, y + 14, 3 * bat, 16), border_radius=3)
        self.text(screen, f"{bat:.1f}%", (x + 306, y + 14))
        mode = fd.get("mode") or "-"
        pygame.draw.rect(screen, MODE_COL.get(mode, (120, 120, 120)), (x + 380, y + 12, 90, 20), border_radius=4)
        self.text(screen, mode.upper(), (x + 388, y + 14), self.font, (20, 20, 20))
        tr, m = fd["truth"], fd["mind"]
        self.text(screen, f"TURN {fd['t']}", (x + 490, y + 14), self.big)
        life = m.get("life") or {}
        est = f"agent's estimate: {life.get('mean', '?')} turns left (10-90%: {life.get('p10', '?')}-{life.get('p90', '?')})," \
              f" p_move {m.get('p_move', 0):.2f}"
        self.text(screen, est, (x, y + 40), self.font)
        if not hidden:
            truth = f"truth: {th['target_name']} energy {tr['energy']:.2f}, ~{tr['remaining_life']:.0f} turns left" + \
                    ("" if tr["alive"] else "  (DEAD)")
            self.text(screen, truth, (x, y + 58), self.font, (255, 150, 200))
        lines = self.wrap("Reasoning: " + (fd.get("reason") or ""), BAR.w - 30)
        for i, l in enumerate(lines[:3]):
            self.text(screen, l, (x, y + 82 + 17 * i), self.font, th["accent"] if i == 0 else th["text"])
        ov = fd.get("override")
        if ov:
            p = ov["plans"]
            msg = f"ENDGAME OVERRIDE  safe {p['safe']:.2f} | dash {p['dash']:.2f} | lean {p['lean']:.2f}  ->  {ov['choice'].upper()}"
            w = self.big.size(msg)[0] + 24
            pygame.draw.rect(screen, (170, 40, 30), (BAR.right - w - 12, BAR.y + 8, w, 32), border_radius=6)
            self.text(screen, msg, (BAR.right - w, BAR.y + 12), self.big, (255, 240, 220))
        help_ = "space play/pause  <-/-> step  +/- speed  B belief  F fog  P path  T truth  M full map  Q quit"
        self.text(screen, extra_help or help_, (x, BAR.bottom - 18), self.small, th["dim"])

    def draw(self, screen, fd: dict, receipts: list[str], tick: int, hidden: bool = False, extra_help: str = ""):
        th = self.th
        screen.fill(th["bg"])
        title = th["title"]
        self.text(screen, title, (12, 8), self.big, th["accent"])
        sub = f"{self.cfg.get('world', {}).get('weather', {}).get('resolved', {}).get('location', '')} weather " \
              f"{fd.get('weather', '')}  seed {self.cfg.get('seed')}  art: {self.sprites.pack}"
        self.text(screen, sub, (14 + self.big.size(title)[0] + 16, 14), self.small, th["dim"])
        self.draw_mind(screen, fd, tick)
        self.draw_truth(screen, fd, tick, hidden)
        self.draw_receipts(screen, receipts)
        self.draw_bar(screen, fd, hidden, extra_help)
        if fd.get("outcome"):
            word = {"rescued": "RESCUED!", "target_died": "TOO LATE...", "battery_depleted": "OUT OF BATTERY",
                    "step_cap": "TIME UP"}.get(fd["outcome"], fd["outcome"].upper())
            col = (90, 230, 120) if fd["outcome"] == "rescued" else (240, 90, 80)
            img = self.huge.render(word, True, col)
            box = img.get_rect(center=(TRUTH.centerx, TRUTH.centery))
            pygame.draw.rect(screen, (0, 0, 0), box.inflate(30, 16), border_radius=8)
            screen.blit(img, box)


def handle_common_keys(viewer: Viewer, key) -> bool:
    """Layer toggles shared by replay and play. Returns True if the key was used."""
    toggles = {pygame.K_b: "belief", pygame.K_f: "fog", pygame.K_p: "path", pygame.K_t: "truth"}
    if key in toggles:
        viewer.layers[toggles[key]] = not viewer.layers[toggles[key]]
        return True
    if key == pygame.K_m:
        viewer.fullmap = not viewer.fullmap
        return True
    return False


def weather_str(w: dict) -> str:
    return f"{w.get('temperature', '?')}C {w.get('humidity', '?')}%RH rain {w.get('rain', 0)}mm"

