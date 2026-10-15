"""Render a saved run to an animated GIF, offscreen (no window)."""
from __future__ import annotations

import os
from pathlib import Path


def export_gif(run_dir, out, every: int | None = None, max_frames: int | None = None, scale: float = 0.6) -> str:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from PIL import Image

    from .replay import RunLog
    from .viewer import H, W, Viewer

    log = RunLog(run_dir)
    viz = log.cfg.get("viz", {})
    every = every or int(viz.get("gif_every", 2))
    max_frames = max_frames or int(viz.get("gif_max_frames", 400))
    pygame.init()
    pygame.display.set_mode((1, 1))
    screen = pygame.Surface((W, H))
    viewer = Viewer(log.cfg, log.walls, log.static)
    viewer.static_ping = True  # fixed frames: each scan's ring at full radius, low opacity
    viewer.set_timeline(*log.timeline())
    idx = list(range(0, len(log), every))
    if len(idx) > max_frames:
        step = len(idx) / max_frames
        idx = [idx[int(k * step)] for k in range(max_frames)]
    if idx[-1] != len(log) - 1:
        idx.append(len(log) - 1)
    frames = []
    size = (int(W * scale), int(H * scale))
    for n, i in enumerate(idx):
        viewer.draw(screen, log.frame(i), log.receipts[:i + 1], tick=n * every)
        img = Image.frombytes("RGB", (W, H), pygame.image.tostring(screen, "RGB")).resize(size, Image.LANCZOS)
        frames.append(img.quantize(colors=255, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    durations = [90] * (len(frames) - 1) + [3000]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    pygame.quit()
    return f"wrote {len(frames)} frames to {out}"
