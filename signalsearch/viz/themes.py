"""Themes are skins only: names, colours and which sprite plays which role. Code stays agent/target."""
from __future__ import annotations

THEMES = {
    "castle": {
        "title": "The Knight and the Lost Princess",
        "target": {"princess": "the princess", "prince": "the prince"},
        "bg": (18, 16, 22), "panel": (30, 27, 36), "accent": (232, 184, 92), "text": (230, 224, 214),
        "dim": (150, 142, 136), "fog": (16, 20, 34), "free": (74, 68, 66), "wall": (150, 140, 140),
        "oob": (6, 6, 8),
    },
    "walle": {
        "title": "WALL-E and EVE's Signal",
        "target": {"princess": "the probe", "prince": "the probe"}, "tint_mult": (255, 215, 160),
        "bg": (24, 18, 12), "panel": (40, 32, 22), "accent": (246, 196, 70), "text": (240, 228, 206),
        "dim": (170, 150, 120), "fog": (30, 24, 18), "free": (104, 86, 60), "wall": (160, 136, 104),
        "oob": (8, 6, 4),
    },
}


def theme(cfg: dict) -> dict:
    t = cfg.get("theme", {})
    name = t.get("name", "castle")
    out = dict(THEMES.get(name, THEMES["castle"]))
    out["name"] = name
    if name == "walle":
        out["rescuer_sprite"], out["target_sprite"] = "robot", "probe"
    else:
        out["rescuer_sprite"] = t.get("rescuer_sprite", "knight_m")
        out["target_sprite"] = t.get("target_sprite", "princess")
    out["target_name"] = out["target"].get(t.get("target_sprite", "princess"), "the target")
    return out
