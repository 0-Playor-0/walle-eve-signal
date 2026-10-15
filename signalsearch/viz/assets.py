"""Art: Kenney Tiny Dungeon (CC0, downloaded automatically) and 0x72 DungeonTileset II (CC0, placed
manually; preferred when present). Anything neither pack covers is drawn procedurally in 16x16 pixel
style, and if no pack is present at all everything is procedural, so the game always runs."""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import pygame
import yaml

KENNEY_URL = "https://opengameart.org/sites/default/files/kenney_tinydungeon.zip"
KENNEY_DIR = "kenney_tiny_dungeon"
X72_DIR = "0x72_DungeonTilesetII_v1.7"
X72_ZIP = "0x72_DungeonTilesetII_v1.7.zip"
TILE = 16


def fetch_kenney(assets_dir: Path) -> str:
    """Download + unzip Kenney Tiny Dungeon into assets/kenney_tiny_dungeon (idempotent)."""
    import requests
    dest = Path(assets_dir) / KENNEY_DIR
    if (dest / "Tiles" / "tile_0000.png").exists():
        return f"already present: {dest}"
    r = requests.get(KENNEY_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for m in z.infolist():
            if m.filename.startswith("__MACOSX") or m.is_dir():
                continue
            target = (dest / m.filename).resolve()
            if dest.resolve() not in target.parents:  # never write outside the asset folder
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
    return f"downloaded Kenney Tiny Dungeon to {dest}"


def _unpack_0x72(assets_dir: Path) -> Path | None:
    folder = assets_dir / X72_DIR
    if (folder / "frames").is_dir():
        return folder
    z = assets_dir / X72_ZIP
    if not z.exists():
        return None
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            if m.filename.startswith("__MACOSX") or m.is_dir() or not m.filename.startswith(X72_DIR):
                continue
            target = (assets_dir / m.filename).resolve()
            if assets_dir.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(m))
    return folder if (folder / "frames").is_dir() else None


# ---- procedural 16x16 sprites ------------------------------------------------------------------
def _surf(w=TILE, h=TILE):
    return pygame.Surface((w, h), pygame.SRCALPHA)


def _px(s, pts, color):
    for x, y in pts:
        s.set_at((x, y), color)


def draw_floor(variant=0, tint=(70, 62, 58)):
    s = _surf()
    s.fill(tint)
    shade = tuple(max(0, c - 10) for c in tint)
    pygame.draw.line(s, shade, (0, 15), (15, 15))
    pygame.draw.line(s, shade, (15, 0), (15, 15))
    if variant % 3 == 1:
        _px(s, [(4, 5), (5, 5), (11, 10)], shade)
    return s


def draw_wall(tint=(118, 108, 112)):
    s = _surf()
    s.fill(tint)
    mortar = tuple(max(0, c - 40) for c in tint)
    light = tuple(min(255, c + 25) for c in tint)
    for y in (0, 5, 10, 15):
        pygame.draw.line(s, mortar, (0, y), (15, y))
    for row, off in enumerate((0, 4, 0)):
        for x in range(off, 16, 8):
            pygame.draw.line(s, mortar, (x, row * 5), (x, row * 5 + 5))
    pygame.draw.line(s, light, (0, 1), (15, 1))
    return s


def draw_dog(frame=0):
    s = _surf()
    body, dark = (150, 98, 54), (96, 60, 32)
    bob = frame % 2
    pygame.draw.rect(s, body, (3, 7 + bob, 9, 5))
    pygame.draw.rect(s, body, (10, 4 + bob, 5, 5))
    _px(s, [(11, 3 + bob), (14, 3 + bob)], dark)               # ears
    _px(s, [(13, 6 + bob)], (20, 20, 20))                        # eye
    _px(s, [(15, 7 + bob)], (20, 20, 20))                        # nose
    legs = [(4, 12), (6, 12), (9, 12), (11, 12)]
    for i, (x, y) in enumerate(legs):
        pygame.draw.line(s, dark, (x, y + bob), (x, y + 2 + ((i + frame) % 2)))
    pygame.draw.line(s, dark, (3, 8 + bob), (1, 5 + bob - frame % 2))  # tail
    return s


def draw_armour():
    s = _surf()
    steel, dark, plume = (170, 176, 188), (90, 96, 110), (180, 40, 40)
    pygame.draw.rect(s, plume, (7, 0, 2, 2))
    pygame.draw.rect(s, steel, (5, 2, 6, 5))
    pygame.draw.line(s, dark, (6, 4), (9, 4))                    # visor slit
    pygame.draw.rect(s, steel, (4, 7, 8, 5))
    pygame.draw.line(s, dark, (8, 7), (8, 11))
    pygame.draw.rect(s, dark, (5, 12, 2, 4))
    pygame.draw.rect(s, dark, (9, 12, 2, 4))
    pygame.draw.line(s, (120, 90, 40), (13, 2), (13, 15))        # spear
    _px(s, [(12, 1), (13, 0), (14, 1)], steel)
    return s


def _flame(s, cx, base, h, frame):
    cols = [(255, 236, 140), (255, 170, 50), (220, 80, 30)]
    sway = [0, 1, 0, -1][frame % 4]
    for i, c in enumerate(reversed(cols)):
        w = 5 - i * 1.5
        top = base - h + i * 2
        pygame.draw.polygon(s, c, [(cx - w, base), (cx + w, base), (cx + sway, top)])


def draw_torch(frame=0):
    s = _surf()
    pygame.draw.rect(s, (90, 70, 50), (7, 9, 2, 7))
    pygame.draw.rect(s, (60, 60, 66), (5, 8, 6, 2))
    _flame(s, 8, 8, 7 + frame % 2, frame)
    return s


def draw_hearth(frame=0):
    s = _surf()
    stone, dark = (126, 116, 110), (70, 62, 60)
    pygame.draw.rect(s, stone, (1, 3, 14, 13))
    pygame.draw.rect(s, dark, (3, 7, 10, 9))
    _flame(s, 8, 15, 6 + frame % 3, frame)
    pygame.draw.rect(s, (160, 150, 140), (0, 2, 16, 2))
    return s


def draw_knight(female=False, frame=0):
    s = _surf()
    steel, dark = (190, 196, 206), (80, 86, 100)
    hair = (200, 160, 60) if female else None
    bob = frame % 2
    if hair:
        pygame.draw.rect(s, hair, (4, 3 + bob, 8, 6))
    pygame.draw.rect(s, steel, (5, 1 + bob, 6, 6))
    pygame.draw.line(s, dark, (6, 4 + bob), (9, 4 + bob))
    pygame.draw.rect(s, (40, 80, 160), (4, 7 + bob, 8, 5))
    pygame.draw.rect(s, dark, (5, 12, 2, 4))
    pygame.draw.rect(s, dark, (9, 12, 2, 4))
    pygame.draw.line(s, (220, 220, 230), (13, 4 + bob), (13, 12 + bob))
    return s


def draw_royal(prince=False, frame=0):
    s = _surf()
    bob = frame % 2
    skin, gold = (240, 200, 170), (240, 200, 60)
    dress = (60, 120, 200) if prince else (230, 110, 170)
    pygame.draw.rect(s, (120, 70, 40) if prince else (240, 210, 90), (5, 2 + bob, 6, 4))
    pygame.draw.rect(s, skin, (6, 3 + bob, 4, 4))
    _px(s, [(5, 1 + bob), (7, 0 + bob), (8, 1 + bob), (10, 1 + bob), (6, 1 + bob), (9, 0 + bob)], gold)
    pygame.draw.polygon(s, dress, [(5, 7 + bob), (10, 7 + bob), (13, 15), (2, 15)])
    return s


def draw_robot(frame=0):
    """A boxy little trash-compactor robot (theme `walle`, original pixel art)."""
    s = _surf()
    body, dark, eye = (214, 170, 60), (90, 70, 40), (60, 60, 70)
    pygame.draw.rect(s, body, (3, 6, 10, 6))
    pygame.draw.rect(s, dark, (2, 12, 12, 3))
    for x in range(3, 14, 3):
        s.set_at((x + frame % 2, 13), (40, 40, 40))
    pygame.draw.line(s, dark, (8, 5), (8, 6))
    pygame.draw.rect(s, eye, (4, 1, 3, 4))
    pygame.draw.rect(s, eye, (9, 1, 3, 4))
    _px(s, [(5, 2), (10, 2)], (200, 220, 255))
    return s


def draw_probe(frame=0):
    """A sleek white probe robot (theme `walle`, original pixel art)."""
    s = _surf()
    shell = (236, 240, 246)
    bob = [0, 1, 1, 0][frame % 4]
    pygame.draw.ellipse(s, shell, (4, 1 + bob, 8, 7))
    pygame.draw.ellipse(s, shell, (4, 7 + bob, 8, 8))
    pygame.draw.rect(s, (20, 20, 30), (5, 3 + bob, 6, 3))
    _px(s, [(6, 4 + bob), (9, 4 + bob)], (80, 180, 255))
    return s


PROCEDURAL = {
    "dog": lambda f: draw_dog(f), "armour": lambda f: draw_armour(), "torch": lambda f: draw_torch(f),
    "hearth": lambda f: draw_hearth(f), "knight_m": lambda f: draw_knight(False, f),
    "knight_f": lambda f: draw_knight(True, f), "princess": lambda f: draw_royal(False, f),
    "prince": lambda f: draw_royal(True, f), "robot": lambda f: draw_robot(f), "probe": lambda f: draw_probe(f),
}


class Sprites:
    """name -> list of frames (pygame Surfaces, native size; scale when drawing)."""

    def __init__(self, assets_dir: Path, pack: str | None = None):
        assets_dir = Path(assets_dir)
        self.frames: dict[str, list[pygame.Surface]] = {}
        mapping = yaml.safe_load((assets_dir / "mapping.yaml").read_text()) if (assets_dir / "mapping.yaml").exists() else {}
        x72 = _unpack_0x72(assets_dir) if pack in (None, "0x72") else None
        kenney = assets_dir / KENNEY_DIR / "Tiles"
        if x72 is not None and "0x72" in mapping:
            self.pack = "0x72"
        elif kenney.is_dir() and "kenney" in mapping and pack in (None, "kenney"):
            self.pack = "kenney"
        else:
            self.pack = "procedural"
        if self.pack != "procedural":
            m = mapping[self.pack]
            base = assets_dir / m["dir"]
            for name, files in m.items():
                if name == "dir":
                    continue
                try:
                    self.frames[name] = [pygame.image.load(str(base / f)) for f in files]
                except (FileNotFoundError, pygame.error):
                    pass
        if "floor" not in self.frames:
            self.frames["floor"] = [draw_floor(i) for i in range(3)]
        if "wall" not in self.frames:
            self.frames["wall"] = [draw_wall()]
        for name, fn in PROCEDURAL.items():
            if name not in self.frames:
                self.frames[name] = [fn(i) for i in range(4)]

    def get(self, name: str, tick: int = 0) -> pygame.Surface:
        fr = self.frames.get(name) or self.frames["floor"]
        return fr[tick % len(fr)]

    def variant(self, name: str, x: int, y: int) -> pygame.Surface:
        fr = self.frames[name]
        return fr[(x * 73856093 ^ y * 19349663) % len(fr)]


def scaled(surface: pygame.Surface, cell: int, cache: dict) -> pygame.Surface:
    """Scale a sprite to cell width (keeping aspect for tall 16x28 characters), cached."""
    key = (id(surface), cell)
    if key not in cache:
        w, h = surface.get_size()
        k = cell / w
        cache[key] = pygame.transform.scale(surface, (cell, max(1, int(math.ceil(h * k)))))
    return cache[key]
