"""Replay a saved run (no re-planning: everything comes from the logs)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pygame
import yaml

from ..sim.logger import load_jsonl
from .viewer import H, W, Viewer, events_of, handle_common_keys, receipt, weather_str

SENSORS = ("lidar", "radar", "thermal", "cv")


class RunLog:
    def __init__(self, run_dir):
        d = Path(run_dir)
        self.dir = d
        self.cfg = yaml.safe_load((d / "config.resolved.yaml").read_text())
        self.static = json.loads((d / "world.json").read_text())
        self.walls = np.load(d / "world.npz")["walls"]
        self.trace = load_jsonl(d / "trace.jsonl")
        self.truth = {r["t"]: r for r in load_jsonl(d / "truth.jsonl")}
        self.snaps = np.load(d / "beliefs.npz")
        self.receipts = [receipt(r["observation"]) for r in self.trace]
        self.weather = weather_str(self.static["weather"])

    def timeline(self) -> tuple[dict[int, list[str]], int]:
        """Event markers for the whole mission (detections, camera labels, charges, overrides) and its length."""
        ev = {}
        for i, r in enumerate(self.trace):
            kinds = events_of(r["observation"], r.get("override"))
            if kinds:
                ev[i] = kinds
        return ev, len(self.trace)

    def __len__(self):
        return len(self.trace)

    def frame(self, i: int) -> dict:
        rec = self.trace[i]
        t = rec["t"]
        obs = rec["observation"]
        key = f"{t:05d}"
        plan = rec.get("plan") or {}
        act = obs["action"].split()[0]
        scan = {"kind": act, "pose": obs["pose"], "facing": obs["facing"]} if act in SENSORS else None
        ep = obs.get("episode", {})
        prev = self.trace[i - 1] if i > 0 else rec
        return {
            "t": t, "battery": obs["battery"], "reason": rec.get("reason"), "mode": rec.get("mode") or prev.get("mode"),
            "override": rec.get("override"), "outcome": ep.get("reason") if ep.get("done") else None,
            "weather": self.weather,
            "mind": {"window": self.snaps["w" + key].tolist(), "occ": self.snaps["o" + key],
                     "belief": self.snaps["b" + key].astype(np.float32), "pose": obs["pose"], "facing": obs["facing"],
                     "hearths": rec.get("hearths") or prev.get("hearths", []), "path": plan.get("path"),
                     "goal": plan.get("route_goal") or plan.get("goal"), "scan": scan, "fused": rec.get("fused", []),
                     "readings": obs.get("readings") or {},
                     "life": rec.get("remaining_life") or prev.get("remaining_life"),
                     "p_move": rec.get("p_move_estimate") or prev.get("p_move_estimate") or 0.0},
            "truth": self.truth[t],
            "trail": [self.truth[r["t"]]["rescuer"] for r in self.trace[max(0, i - 600):i + 1]],
        }


def replay(run_dir, gif: str | None = None) -> None:
    if gif:
        from .gif_export import export_gif
        print(export_gif(run_dir, gif))
        return
    log = RunLog(run_dir)
    pygame.init()
    screen = pygame.display.set_mode((W, H), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption(f"Replay - {Path(run_dir).name}")
    viewer = Viewer(log.cfg, log.walls, log.static)
    viewer.set_timeline(*log.timeline())
    clock = pygame.time.Clock()
    i, playing, tps, acc, tick = 0, True, float(log.cfg.get("viz", {}).get("turns_per_second", 8)), 0.0, 0
    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        tick += 1
        for ev in pygame.event.get():  # every frame, or macOS never shows the window
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                j = viewer.timeline_turn(ev.pos)
                if j is not None:
                    i, playing = min(j, len(log) - 1), False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_RIGHT:
                    i, playing = min(i + 1, len(log) - 1), False
                elif ev.key == pygame.K_LEFT:
                    i, playing = max(i - 1, 0), False
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    tps = min(tps * 1.5, 120)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    tps = max(tps / 1.5, 0.5)
                elif ev.key == pygame.K_HOME:
                    i = 0
                else:
                    handle_common_keys(viewer, ev.key)
        if playing:
            acc += dt * tps
            while acc >= 1:
                acc -= 1
                i = min(i + 1, len(log) - 1)
        viewer.draw(screen, log.frame(i), log.receipts[:i + 1], tick,
                    extra_help=f"space play/pause  <-/-> step  +/- speed ({tps:.1f} t/s)  click timeline: jump  B belief  "
                               f"F fog  P path  T truth  M full map  L legend  Q quit")
        pygame.display.flip()
    pygame.quit()
