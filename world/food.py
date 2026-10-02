"""Еда в мире: ягоды на кустах, плоды на деревьях, грибы. Какие они на вид - из сети мира
(это её кусты, деревья и грибы, их цвет и размер); какие они на вкус - из сети еды по зерну
мира: одни вкусные, другие нет. Он этого не знает - узнаёт, попробовав.

Сеть еды: маленькая CPPN, одна на все миры; вход - зерно мира и признаки еды (вид, цвет,
размер), выход - вкус -1..1. Её веса отобраны, как генератор тела: из 200 случайных сетей
взята та, у которой на 12 мирах вкусной выходит около половины еды (0.52; в каждом мире от
22 до 80 %) и вкусы разные (разброс 0.6). Первая попавшаяся сеть делала вкусным всё.
Съеденное отрастает за REGROW с.
"""
import math

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

EDIBLE = {"bush": "ягоды", "tree": "плоды", "mushroom": "грибы"}
KINDS = tuple(EDIBLE)
N_FEAT = len(KINDS) + 5
LAYERS = (N_FEAT + Z_DIM, 12, 1)
REGROW = 180.0               # с - съеденное отрастает
NET_SEED = 7                 # отобранная сеть (см. выше)
PORTION = 0.45               # столько голода уходит за один раз


def features(kind, color, size):
    """Признаки еды: вид (один из трёх), оттенок (cos, sin), насыщенность, яркость, размер."""
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(*color)
    f = np.zeros(N_FEAT, np.float32)
    f[KINDS.index(kind)] = 1.0
    k = len(KINDS)
    f[k], f[k + 1] = math.cos(2 * math.pi * h), math.sin(2 * math.pi * h)
    f[k + 2], f[k + 3], f[k + 4] = s, v, size
    return f


def _net():
    rng = np.random.default_rng(NET_SEED)
    ws, bs = [], []
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.2 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.3, b))
    return ws, bs


_NET = _net()


def taste(world_seed, feats):
    """Вкус -1 (невкусно) .. 1 (вкусно) - сеть еды."""
    ws, bs = _NET
    # признаки еды весят больше зерна мира: иначе в мире всё вкусное или всё невкусное
    h = np.concatenate([2.0 * np.asarray(feats, float), 0.3 * seed_vector(world_seed)])[None]
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    return float(np.tanh(1.5 * h[0, 0]))


class Food:
    """Еда мира и что с ней сейчас: какие места съедены и когда отрастут."""

    def __init__(self, world):
        self.world = world
        self.items = [(i, e) for i, e in enumerate(world.entities) if e.kind in EDIBLE]
        self.feats = {i: features(e.kind, e.color, e.size) for i, e in self.items}
        self.eaten = {}              # индекс -> когда отрастёт

    def available(self, t):
        return [(i, e) for i, e in self.items if self.eaten.get(i, -1.0) <= t]

    def eat(self, i, t):
        e = self.world.entities[i]
        self.eaten[i] = t + REGROW
        return taste(self.world.seed, self.feats[i]), e
