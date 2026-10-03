"""Вид и порода птиц мира - выход маленькой сети птиц по зерну мира: размер, окрас, форма крыла
и порода - мелкие быстрые (часто машут) или крупные парители (машут редко, держатся на крыле)."""
import colorsys
import math
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

LAYERS = (Z_DIM, 10, 6)


def _net():
    rng = np.random.default_rng(0xB1D)
    ws, bs = [], []
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.3 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_NET = _net()


@dataclass
class BirdLook:
    size: float          # 0..1
    color: tuple
    wing: float          # 0 короткое .. 1 длинное
    glider: bool         # парит (крупная) или машет (мелкая)

    @property
    def span(self):
        """Размах крыльев, м."""
        return 0.25 + 0.9 * self.size if self.glider else 0.18 + 0.25 * self.size


def look(world_seed):
    h = seed_vector(int(world_seed) * 29 + 11)[None]
    ws, bs = _NET
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    o = 1.0 / (1.0 + np.exp(-1.6 * h[0]))
    color = colorsys.hsv_to_rgb(float(o[1]), 0.3 + 0.6 * float(o[2]), 0.3 + 0.6 * float(o[3]))
    return BirdLook(size=float(o[0]), color=tuple(round(c, 4) for c in color), wing=float(o[4]), glider=bool(o[5] > 0.8))
