"""Горы больших миров (спецификация 2.0, раздел 8a): вершины 60-100 м над макро-рельефом.

У каждой горы:
- **лёгкий путь** - пологий склон, средний уклон EASY (замер: по 0.2 он идёт так же быстро, как
  по ровному, 0.6 м/с; по 0.3 - 0.4 м/с; по 0.4 застревает): по нему гору проходят все, закон
  «ловушек нет» держится через него;
- **стена** - склон FACE (м на м) - круче его шага (body/physics.WALL_SLOPE 2.5): пройти нельзя,
  только залезть (skills/climb.py);
- плоская **вершина** TOP м - для флага и радости.
На склонах горы холмы сети рельефа приглушены (HILL_DAMP), иначе лёгкий путь местами круче 0.4.

Сколько гор, какой высоты, где и с какой стороны стена - выход маленькой сети гор по зерну
мира (как вода и макро-рельеф), веса - эволюция (train/mountains.py, models/mountains.npz) на
«в каждом мире 1280 м есть гора, высоты разные, стены с обеих сторон, всё проходимо». Только
для 640 и 1280 м: прежние размеры - те же байты.
"""
import math
import os
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, Z_DIM, seed_vector

LAYERS = (Z_DIM, 12, 10)
SIZE = sum(a * b + b for a, b in zip(LAYERS, LAYERS[1:]))
EASY = 0.22                  # средний уклон лёгкого пути
FACE = 3.3                   # уклон стены
HEIGHT = (60.0, 100.0)
TOP = 3.0                    # м - плоская вершина
GAP = 60.0                   # м - между горами хотя бы столько равнины
HILL_DAMP = 0.97             # на горе холмы и макро-рельеф - 3 % от прежних (замер: с 8 % лёгкий
                             # путь местами 0.9, стена местами 1.2 - круче шага и положе стены)
RAMP = 30.0                  # м - от следа горы к равнине холмы возвращаются
BASE = 80.0                  # м - медленный рельеф, на котором стоит гора
MAX_POND = 3.0               # м - в мире с горами вода не глубже (котловина у подножия - не озеро в 60 м)
MIN_WIDTH = 640.0


@dataclass
class Mountain:
    summit: float            # x середины вершины
    height: float            # м над подножием
    steep: int               # +1 - стена справа от вершины, -1 - слева
    snow: float              # 0..1 - снег на вершине (только вид)
    top: float = 0.0         # м - высота вершины над уровнем мира (ставит apply)

    @property
    def easy_len(self):
        return self.height / EASY

    @property
    def face_len(self):
        return self.height / FACE

    def face_foot(self):
        """x подножия стены."""
        return self.summit + self.steep * (TOP / 2 + self.face_len)

    def easy_foot(self):
        """x начала лёгкого пути."""
        return self.summit - self.steep * (TOP / 2 + self.easy_len)

    def footprint(self):
        return self.easy_len + TOP + self.face_len


def _net(genome):
    ws, bs, k = [], [], 0
    for a, b in zip(LAYERS, LAYERS[1:]):
        ws.append(genome[k:k + a * b].reshape(a, b))
        k += a * b
        bs.append(genome[k:k + b])
        k += b
    return ws, bs


def outputs(genome, seed):
    h = seed_vector(int(seed) * 17 + 3)[None]
    ws, bs = _net(genome)
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    return 1.0 / (1.0 + np.exp(-h[0]))


_default = None


def default_genome():
    global _default
    if _default is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mountains.npz")
        try:
            with np.load(path) as data:
                g = np.asarray(data["genome"], float)
            if g.shape != (SIZE,) or not np.isfinite(g).all():
                raise ValueError("чужой файл")
        except Exception:  # noqa: BLE001
            g = np.random.default_rng(2).normal(0.0, 1.0, SIZE)
        _default = g
    return _default


def plan(width, seed, genome=None):
    """Горы мира: [Mountain] (пусто для миров меньше 640 м)."""
    width = float(width)
    if width < MIN_WIDTH:
        return []
    o = outputs(default_genome() if genome is None else genome, seed)
    if width >= 1280:
        count = 1 + int(o[0] > 0.5)
    else:
        count = int(o[0] > 0.4)
    peaks = []
    # стены всех гор мира - в одну сторону: две стены лицом друг к другу - долина-ловушка
    # (замер: из неё не выйти ни шагом, ни прыжком); в разных мирах - в разные стороны
    steep = 1 if o[2] > 0.5 else -1
    for k in range(count):
        height = HEIGHT[0] + (HEIGHT[1] - HEIGHT[0]) * float(o[1 + 3 * k])
        peaks.append(Mountain(0.0, height, steep, float(o[9])))
    # раскладка по кольцу: подножия не заходят друг на друга, между ними не меньше GAP
    need = sum(m.footprint() + GAP for m in peaks)
    while peaks and need > width:
        peaks.pop()
        need = sum(m.footprint() + GAP for m in peaks)
    if not peaks:
        return []
    slack = width - need
    x = width * float(o[3])
    for k, m in enumerate(peaks):
        # вершина - на своём месте внутри следа горы (лёгкий путь со стороны -steep)
        left = x
        m.summit = (left + (m.easy_len + TOP / 2 if m.steep > 0 else m.face_len + TOP / 2)) % width
        extra = slack * (float(o[6]) if k == 0 else 0.0)
        x = left + m.footprint() + GAP + extra
    return peaks


def local(m, xs, width):
    """Координата вдоль горы: 0 - вершина, плюс - к стене. Лёгкий путь бывает длиннее половины
    кольца (82 м в мире 640 - 375 м), поэтому кратчайшее расстояние до вершины не годится: на
    полкольце от вершины профиль прыгал на 13 м (замер: ловушки) - берётся тот обход, что
    попадает в след горы."""
    xs = np.asarray(xs, float)
    d1 = (xs - m.summit) % width
    s1, s2 = d1 * m.steep, (d1 - width) * m.steep
    lo, hi = -(TOP / 2 + m.easy_len), TOP / 2 + m.face_len

    def out(v):
        return np.maximum(lo - v, v - hi)
    return np.where(out(s1) <= out(s2), s1, s2)


def profile(peaks, width, dx, n):
    """(высоты гор на отсчётах, «на горе» 0..1: 1 на всём следе горы, к равнине - за RAMP м)."""
    xs = np.arange(n) * dx
    mount = np.zeros(n)
    on = np.zeros(n)
    for m in peaks:
        s = local(m, xs, width)              # плюс - к стене
        top = np.abs(s) <= TOP / 2
        face = s > TOP / 2
        easy = s < -TOP / 2
        h = np.zeros(n)
        h[top] = m.height
        h[face] = np.clip(m.height - FACE * (s[face] - TOP / 2), 0.0, None)
        u = np.clip(1.0 - (-s[easy] - TOP / 2) / m.easy_len, 0.0, 1.0)
        h[easy] = m.height * u
        mount = np.maximum(mount, h)
        lo, hi = -(TOP / 2 + m.easy_len), TOP / 2 + m.face_len
        out = np.maximum(lo - s, s - hi)               # сколько м за следом горы
        on = np.maximum(on, np.clip(1.0 - out / RAMP, 0.0, 1.0))
    return mount, on


def smooth_ring(h, dx, band):
    n = max(1, int(band / dx))
    pad = np.concatenate([h[-n:], h, h[:n]])
    return np.convolve(pad, np.ones(n) / n, mode="same")[n:-n]


def apply(heights, peaks, width, dx):
    """Высоты мира с горами и доля «на горе». На горе земля - выше из двух: рельеф (холмы
    приглушены, медленный рельеф за BASE м остаётся) и линия горы: вершина на H над рельефом у
    начала лёгкого пути, вниз - лёгкий путь с уклоном EASY и стена с уклоном FACE, до встречи с
    рельефом. Так лёгкий путь - ровно EASY, что бы ни делал макро-рельеф (замер: гора, просто
    положенная сверху, давала лёгкий путь 0.36 в среднем и до 0.47, и ловушки у подножия)."""
    if not peaks:
        return heights, np.zeros(len(heights))
    n = len(heights)
    xs = np.arange(n) * dx
    _, on = profile(peaks, width, dx, n)
    base = smooth_ring(np.asarray(heights, float), dx, BASE)
    out = base + (heights - base) * (1.0 - HILL_DAMP * on)
    for m in peaks:
        s = local(m, xs, width)
        foot = int(round((m.easy_foot() % width) / dx)) % n
        top = base[foot] + m.height
        m.top = float(top)
        line = np.where(s < 0, top - EASY * np.maximum(0.0, -s - TOP / 2), top - FACE * np.maximum(0.0, s - TOP / 2))
        inside = (s >= -(TOP / 2 + m.easy_len)) & (s <= TOP / 2 + m.face_len + RAMP)
        out = np.where(inside, np.maximum(out, line), out)
    return out, on


def summit_at(peaks, x, width, near=TOP / 2 + 0.3):
    """Гора, на вершине которой точка x (или None)."""
    for m in peaks:
        if abs((x - m.summit + width / 2) % width - width / 2) <= near:
            return m
    return None


def on_face(peaks, x, width):
    """Гора, у стены которой (на самой стене) точка x, или None."""
    for m in peaks:
        s = float(local(m, x, width))
        if TOP / 2 < s < TOP / 2 + m.face_len:
            return m
    return None


def easy_slope_max(world, m, window=2.0):
    """Самый крутой кусок лёгкого пути (уклон на window м) - закон «пройти можно»."""
    xs = np.arange(0.0, m.easy_len - 1.0, 0.5)
    pts = [m.summit - m.steep * (TOP / 2 + 0.5 + s) for s in xs]
    worst = 0.0
    for x in pts:
        worst = max(worst, abs(world.height_at(x + window / 2) - world.height_at(x - window / 2)) / window)
    return worst


def face_wall(world, m, step=0.25):
    """Стена горы как она есть: от края вершины вниз, пока уклон не меньше WALL (её шагом не
    пройти). Возвращает (высота стены, самый пологий её кусок на 0.5 м)."""
    from body.physics import WALL_SLOPE
    s, drop, flattest = TOP / 2 + 0.25, 0.0, math.inf
    while s < TOP / 2 + m.face_len + 5.0:
        x = m.summit + m.steep * s
        k = abs(world.height_at(x + 0.25) - world.height_at(x - 0.25)) / 0.5
        if k < WALL_SLOPE:
            break
        flattest = min(flattest, k)
        drop += k * step
        s += step
    return drop, flattest
