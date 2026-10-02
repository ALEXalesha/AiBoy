"""Вид и нрав собаки - выход маленькой сети собак по зерну мира (не правило и не таблица):

- вид: размер (от маленькой до большой), окрас (основной и пятна), уши (торчком - висячие),
  хвост (короткий - длинный, прямой - бубликом);
- нрав: пугливость, игривость, выносливость (0..1).

В одном мире собака всегда одна и та же; в другом - другая.
"""
import colorsys
import math
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

LAYERS = (Z_DIM, 12, 10)


def _net():
    rng = np.random.default_rng(0xD06)
    ws, bs = [], []
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.3 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_NET = _net()


@dataclass
class DogLook:
    size: float          # 0..1
    color: tuple
    spots: tuple
    ears: float          # 0 торчком .. 1 висячие
    tail: float          # 0 короткий .. 1 длинный
    curl: float          # 0 прямой .. 1 бубликом
    shy: float
    playful: float
    stamina: float

    @property
    def length(self):
        """Длина туловища, м: от 0.45 (маленькая) до 0.85 (большая)."""
        return 0.45 + 0.4 * self.size


def look(world_seed):
    h = seed_vector(int(world_seed) * 13 + 5)[None]
    ws, bs = _NET
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    o = 1.0 / (1.0 + np.exp(-1.6 * h[0]))
    # окрас собак - рыжие, бурые, серые, чёрные, белые: оттенок в тёплой части круга, разная яркость
    hue = 0.02 + 0.1 * float(o[1])
    color = colorsys.hsv_to_rgb(hue, 0.15 + 0.55 * float(o[2]), 0.2 + 0.75 * float(o[3]))
    spots = colorsys.hsv_to_rgb(hue, 0.1 + 0.3 * float(o[2]), 0.95 - 0.7 * float(o[3]))
    return DogLook(size=float(o[0]), color=tuple(round(c, 4) for c in color), spots=tuple(round(c, 4) for c in spots),
                   ears=float(o[4]), tail=float(o[5]), curl=float(o[6]),
                   shy=float(o[7]), playful=float(o[8]), stamina=float(o[9]))
