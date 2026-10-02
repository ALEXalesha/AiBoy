"""Ветер мира: маленькая сеть погоды по зерну мира даёт основную силу и направление ветра и
порывы (три медленных волны своей частоты и силы). Ветер двигает пузыри, самолётик,
бумеранг, птиц и облака.

at(x, y, t) - горизонтальный ветер, м/с (плюс - вправо); выше над землёй чуть сильнее.
"""
import math

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

LAYERS = (Z_DIM, 8, 8)


def _net():
    rng = np.random.default_rng(0x3E7)
    ws, bs = [], []
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.0 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_NET = _net()


class Wind:
    def __init__(self, world_seed):
        h = seed_vector(world_seed)[None]
        ws, bs = _NET
        for i, (w, b) in enumerate(zip(ws, bs)):
            h = h @ w + b
            if i < len(ws) - 1:
                out = np.empty_like(h)
                for j in range(h.shape[1]):
                    out[:, j] = ACTS[j % len(ACTS)](h[:, j])
                h = out
        o = 1.0 / (1.0 + np.exp(-h[0]))
        self.base = 3.5 * o[0] * (1 if o[1] > 0.5 else -1)       # м/с
        self.gust = 0.4 + 1.6 * o[2]
        self.periods = (8 + 20 * o[3], 21 + 30 * o[4], 47 + 40 * o[5])
        self.phases = (6.28 * o[6], 6.28 * o[7], 6.28 * o[3] * o[4])

    def at(self, x, y, t):
        g = sum(math.sin(2 * math.pi * t / p + ph + 0.05 * x) for p, ph in zip(self.periods, self.phases)) / 3
        height = 1.0 + 0.08 * max(0.0, min(y, 10.0))
        return (self.base + self.gust * g) * height
