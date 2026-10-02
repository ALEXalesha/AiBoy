"""Рыбы в прудах. Как выглядит рыба - из сети рыб по зерну водоёма (зерно мира и номер
пруда): длина, высота тела, хвост, цвет. Плавают просто: каждая держит свою глубину и
ходит от берега до берега с покачиванием; всплеск рядом - уплывает от него быстрее.

Клюёт (для удочки, задача 17): вероятность за секунду - от времени суток (на рассвете и
закате рыба активнее, ночью вяло) и от того, насколько глубина наживки близка к глубине рыбы.
"""
import math
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

LAYERS = (Z_DIM + 2, 10, 5)
DAY = 600.0                  # с - сутки по часам рыбы (для клёва)
BITE = 0.08                  # за секунду при лучших условиях
SPEED = 0.35                 # м/с
FLEE = 1.2                   # м/с, если рядом всплеск


def _net():
    rng = np.random.default_rng(0xF15)
    ws, bs = [], []
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.0 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_NET = _net()


def look(world_seed, pond_index, k):
    """(длина м, высота м, хвост 0..1, цвет rgb) рыбы k пруда pond_index - выход сети рыб."""
    import colorsys
    z = seed_vector(world_seed * 7919 + pond_index)
    h = np.concatenate([z, [3.0 * math.sin(k * 1.7), 3.0 * math.cos(k * 1.3)]])[None]   # рыбы одного пруда - разные
    ws, bs = _NET
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    o = 1.0 / (1.0 + np.exp(-h[0]))
    length = 0.12 + 0.25 * o[0]
    height = length * (0.25 + 0.25 * o[1])
    color = colorsys.hsv_to_rgb(float(o[3]), 0.4 + 0.5 * o[4], 0.55 + 0.4 * o[1])
    return float(length), float(height), float(o[2]), tuple(round(c, 4) for c in color)


@dataclass
class Fish:
    pond: int
    x: float
    depth: float                 # доля глубины пруда, где держится (0 - у поверхности)
    dir: int
    length: float
    height: float
    tail: float
    color: tuple
    phase: float = 0.0


class School:
    """Все рыбы мира и как они плавают."""

    def __init__(self, world, water):
        self.world, self.water = world, water
        self.fish = []
        rng = np.random.default_rng([int(world.seed), 0xF1])
        for i, p in enumerate(water.ponds):
            span = (p.right - p.left) % world.width
            n = int(min(6, span / 3.0)) if p.depth > 0.3 else 0
            for k in range(n):
                length, height, tail, color = look(world.seed, i, k)
                x = p.left + span * (0.15 + 0.7 * rng.random())
                self.fish.append(Fish(i, x % world.width, 0.3 + 0.5 * rng.random(), int(rng.choice([-1, 1])),
                                      length, height, tail, color, float(rng.random() * 6.28)))

    def position(self, f):
        """(x, y) рыбы: на своей глубине, с покачиванием."""
        p = self.water.ponds[f.pond]
        floor = self.world.height_at(f.x)
        y = p.level - f.depth * max(0.05, p.level - floor) + 0.04 * math.sin(f.phase)
        return f.x, min(p.level - 0.05, max(floor + f.height, y))

    def step(self, dt, splash_x=None):
        w = self.world.width
        for f in self.fish:
            p = self.water.ponds[f.pond]
            speed = SPEED
            if splash_x is not None:
                d = (f.x - splash_x + w / 2) % w - w / 2
                if abs(d) < 2.0:
                    f.dir = 1 if d >= 0 else -1          # от всплеска
                    speed = FLEE
            nx = f.x + f.dir * speed * dt
            if self.water.depth_at(nx) < 0.12 + f.height:
                f.dir = -f.dir                            # берег - разворот
            else:
                f.x = nx % w
            f.phase += dt * (3.0 + 4.0 * speed)

    def bite_chance(self, t, bait_depth, fish, dt):
        """Вероятность клюнуть за dt: время суток (рассвет и закат - лучше) и глубина наживки."""
        day = (t % DAY) / DAY
        dusk = max(math.exp(-((day - 0.25) / 0.08) ** 2), math.exp(-((day - 0.75) / 0.08) ** 2))
        active = 0.35 + 0.65 * dusk
        if self.world.night:
            active *= 0.5
        match = math.exp(-((bait_depth - fish.depth) / 0.2) ** 2)
        return 1.0 - math.exp(-BITE * active * match * dt)
