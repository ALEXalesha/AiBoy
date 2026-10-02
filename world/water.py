"""Вода в мире: пруды и озёра в котловинах рельефа.

Где вода - решает маленькая сеть воды по зерну мира (веса - эволюция, train/water.py, файл
models/water.npz): вход - зерно мира и признаки котловины (глубина до перелива, ширина, где
она на кольце), выход - есть ли вода в мире вообще и до какой доли глубины налита эта
котловина. Код только находит котловины и их уровень перелива: вода не бывает выше
перелива (иначе потекла бы через край), ручьёв нет (сбоку они читаются как канава).

Защита ходьбы (как защита от ям-ловушек у рельефа): глубоких водоёмов - глубже, чем вброд, -
не больше одного. Мир - кольцо: один глубокий пруд он обходит другой дорогой, два разрезали бы
мир на части, куда не попасть, пока не научится плавать. Лишние глубокие делаются мельче - до
брода.
"""
import math
import os
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

MIN_DEPTH = 0.25             # м - мельче лужа не считается
MAX_SPAN = 60.0              # м - котловина шире не заливается (это уже долина)
WADE = 0.45                  # м - глубже - не вброд (половина ноги взрослого человечка)
N_FEAT = 5
LAYERS = (N_FEAT + Z_DIM, 12, 2)


@dataclass
class Pond:
    left: float                  # м, где вода начинается (может быть больше right - через шов)
    right: float
    level: float                 # м, высота поверхности
    bottom: float                # м, самое глубокое место
    spill: float                 # м, уровень перелива котловины

    @property
    def depth(self):
        return self.level - self.bottom


def basins(heights, dx, width):
    """Котловины рельефа: [(индекс дна, индекс слева, индекс справа, уровень перелива)]. Вода
    поднимается от дна, захватывая соседей, пока следующий сосед не ниже уровня - там перелив."""
    h = np.asarray(heights, float)
    n = len(h)
    mins = [i for i in range(n) if h[i] < h[i - 1] and h[i] <= h[(i + 1) % n]]
    out, seen = [], set()
    max_span = int(MAX_SPAN / dx)
    for i in mins:
        lo = hi = i
        level = h[i]
        while hi - lo < max_span:
            left, right = h[(lo - 1) % n], h[(hi + 1) % n]
            nxt = min(left, right)
            if nxt < level:
                break                            # дальше вниз - перелив
            level = nxt
            if left <= right:
                lo -= 1
            else:
                hi += 1
        if hi - lo >= max_span or level - h[i] < MIN_DEPTH:
            continue
        key = (lo % n, hi % n)
        if key in seen:
            continue
        seen.add(key)
        out.append((i, lo, hi, level))
    return out


def _net(genome):
    ws, bs, k = [], [], 0
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(genome[k:k + a * b].reshape(a, b))
        k += a * b
        bs.append(genome[k:k + b])
        k += b
    return ws, bs


SIZE = sum(a * b + b for a, b in zip(LAYERS, LAYERS[1:]))


def forward(genome, x):
    ws, bs = _net(genome)
    h = x
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    return h


_default = None


def default_genome():
    global _default
    if _default is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "water.npz")
        try:
            with np.load(path) as data:
                g = np.asarray(data["genome"], float)
            if g.shape != (SIZE,) or not np.isfinite(g).all():
                raise ValueError
        except Exception:  # noqa: BLE001
            g = np.random.default_rng(1).normal(0.0, 1.0, SIZE)
        _default = g
    return _default


def ponds(world, genome=None):
    """Водоёмы мира по сети воды."""
    genome = default_genome() if genome is None else genome
    h = world.heights
    dx, width = world.dx, world.width
    n = len(h)
    found = basins(h, dx, width)
    if not found:
        return []
    z = seed_vector(world.seed)
    feats = []
    for i, lo, hi, spill in found:
        phase = 2 * math.pi * i / n
        feats.append([(spill - h[i]) / 3.0, (hi - lo) * dx / 30.0, math.sin(phase), math.cos(phase), 1.0])
    x = np.concatenate([np.array(feats), np.broadcast_to(z, (len(feats), Z_DIM))], axis=1)
    out = forward(genome, x)
    wet = float(np.mean(out[:, 0]))              # есть ли вода в мире: одно решение на мир
    if wet <= 0:
        return []
    cand = []
    for (i, lo, hi, spill), o in zip(found, out):
        frac = 1.0 / (1.0 + math.exp(-o[1]))
        if frac < 0.2:
            continue
        level = h[i] + frac * (spill - h[i])
        if level - h[i] >= MIN_DEPTH:
            cand.append([i, level, spill])
    # защита: глубоких (не вброд) - не больше одного; остальные мельче, до брода
    deep = sorted([c for c in cand if c[1] - h[c[0]] > WADE], key=lambda c: -(c[1] - h[c[0]]))
    for c in deep[1:]:
        c[1] = h[c[0]] + WADE * 0.9
    res = []
    for i, level, spill in cand:
        a, b = i, i
        while h[(a - 1) % n] < level and i - a < n:
            a -= 1
        while h[(b + 1) % n] < level and b - i < n:
            b += 1
        # берег - где рельеф пересекает уровень (между отсчётами - по прямой)
        ha, hb = h[(a - 1) % n], h[a % n]
        left = (a - 1 + (ha - level) / (ha - hb)) * dx if ha != hb else a * dx
        hc, hd = h[b % n], h[(b + 1) % n]
        right = (b + (level - hc) / (hd - hc)) * dx if hd != hc else b * dx
        res.append(Pond(left % width, right % width, float(level), float(h[i]), float(spill)))
    return res


class Water:
    """Вода мира: где она и какой глубины. x - метры мира (кольцо)."""

    def __init__(self, world, pond_list=None):
        self.world = world
        self.ponds = pond_list if pond_list is not None else ponds(world)

    def pond_at(self, x):
        x = x % self.world.width
        for p in self.ponds:
            inside = p.left <= x <= p.right if p.left <= p.right else (x >= p.left or x <= p.right)
            if inside and self.world.height_at(x) < p.level:
                return p
        return None

    def level_at(self, x):
        p = self.pond_at(x)
        return None if p is None else p.level

    def depth_at(self, x):
        p = self.pond_at(x)
        return 0.0 if p is None else max(0.0, p.level - self.world.height_at(x))

    def shores(self):
        """Точки берега, откуда можно пить: края каждого водоёма."""
        return [x for p in self.ponds for x in (p.left, p.right)]
