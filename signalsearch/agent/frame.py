"""Agent frame (199x199 for a 100x100 map, spawn at (99, 99)) and position pruning."""
from __future__ import annotations

import numpy as np


class Frame:
    def __init__(self, map_size: int = 100):
        self.map_size = map_size
        self.size = 2 * map_size - 1
        self.spawn = (map_size - 1, map_size - 1)
        self.xmin = self.xmax = self.spawn[0]
        self.ymin = self.ymax = self.spawn[1]

    def note_inside(self, cells) -> bool:
        """Record cells known to be inside the map. Returns True if the extents changed."""
        cells = np.asarray(cells).reshape(-1, 2)
        if len(cells) == 0:
            return False
        old = (self.xmin, self.xmax, self.ymin, self.ymax)
        self.xmin = min(self.xmin, int(cells[:, 0].min()))
        self.xmax = max(self.xmax, int(cells[:, 0].max()))
        self.ymin = min(self.ymin, int(cells[:, 1].min()))
        self.ymax = max(self.ymax, int(cells[:, 1].max()))
        return old != (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def window(self) -> tuple[int, int, int, int]:
        """Inclusive (x_lo, x_hi, y_lo, y_hi): the real map must lie inside it."""
        n = self.map_size - 1
        return (max(0, self.xmax - n), min(self.size - 1, self.xmin + n),
                max(0, self.ymax - n), min(self.size - 1, self.ymin + n))

    def mask(self) -> np.ndarray:
        x0, x1, y0, y1 = self.window
        m = np.zeros((self.size, self.size), dtype=bool)
        m[x0:x1 + 1, y0:y1 + 1] = True
        return m

    @property
    def resolved(self) -> bool:
        """True when the map offset is fully known (window exactly map-sized in both axes)."""
        x0, x1, y0, y1 = self.window
        return x1 - x0 + 1 == self.map_size and y1 - y0 + 1 == self.map_size

    def to_map(self, p) -> list[int] | None:
        """Agent-frame cell -> true map cell, only once the offset is resolved."""
        if not self.resolved:
            return None
        x0, _, y0, _ = self.window
        return [int(p[0]) - x0, int(p[1]) - y0]
