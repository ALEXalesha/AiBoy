"""Фломастер: рисует. Рисунок - череда штрихов, их выдаёт маленькая рекуррентная сеть-рука (как
декодер sketch-rnn, только крошечная) по коду рисунка: на каждом шаге - куда вести руку (dx,
dy) и поднять ли фломастер. Что рисовать (код) решает голова, учится REINFORCE по красоте.

Красота - его вкус к рисункам: маленькая логистическая сеть по признакам рисунка (гладкость,
замкнутость, симметрия, размах, сколько штрихов), учится на 👍/👎 владельца, как вкус к
музыке 1.1.0; пока оценок нет - врождённое: гладкий и замкнутый рисунок нравится больше
каракулей. Рисунки остаются в мире, где нарисованы, пока мир не сменится, и все попадают в
«Галерею» («Рисунки»). Звук: шорох фломастера.
"""
import math

import numpy as np

from nn import io
from nn.layers import Dense, Sequential, Tanh
from nn.optim import Adam, clip_grads

STEPS = 24               # движений руки на рисунок
SCALE = 0.06             # м за шаг при выходе 1
HID = 24
LATENT = 6
FEATS = 6


def features(points, pens):
    """Признаки рисунка: гладкость, замкнутость, симметрия, размах, самопересечения, длина."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return np.zeros(FEATS)
    d = np.diff(p, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
    smooth = 1.0 - float(np.mean(turn)) / math.pi
    span = float(np.ptp(p[:, 0]) + np.ptp(p[:, 1])) + 1e-6
    closed = 1.0 - min(1.0, float(np.linalg.norm(p[-1] - p[0])) / span)
    c = p.mean(axis=0)
    mirror = p.copy()
    mirror[:, 0] = 2 * c[0] - mirror[:, 0]
    sym = 1.0 - min(1.0, float(np.mean(np.min(np.linalg.norm(p[:, None] - mirror[None], axis=2), axis=1))) / span)
    cross = crossings(p)
    return np.array([smooth, closed, sym, min(1.0, span / 0.6), min(1.0, cross / 6.0),
                     min(1.0, float(np.sum(np.linalg.norm(d, axis=1))) / 2.0)])


def crossings(p):
    """Сколько раз линия пересекает саму себя."""
    def cr(u, v):
        return u[0] * v[1] - u[1] * v[0]
    n = 0
    segs = list(zip(p[:-1], p[1:]))
    for i in range(len(segs)):
        a, b = segs[i]
        for j in range(i + 2, len(segs)):
            if i == 0 and j == len(segs) - 1:
                continue                         # замкнулась - это не пересечение
            c, d = segs[j]
            d1 = cr(b - a, c - a)
            d2 = cr(b - a, d - a)
            d3 = cr(d - c, a - c)
            d4 = cr(d - c, b - c)
            if d1 * d2 < 0 and d3 * d4 < 0:
                n += 1
    return n


class DrawTaste:
    """Вкус к рисункам: врождённое плюс выученное на оценках владельца."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.net = Sequential([Dense(FEATS, 1, rng, np.float64, gain=0.1)])
        self.opt = Adam(self.net.params(), lr=0.05)
        self.ratings = []

    def innate(self, f):
        # гладкая замкнутая линия без самопересечений и не крошечная; каракуля с петлями - нет
        return (0.3 * f[1] + 0.35 * f[0] + 0.35 * (1.0 - f[4])) * (0.4 + 0.6 * f[3])

    def beauty(self, f):
        learned = float(np.tanh(self.net.forward(np.asarray(f)[None])[0, 0])) if self.ratings else 0.0
        return float(np.clip(self.innate(f) + 0.5 * learned, 0.0, 1.0))

    def rate(self, f, rating):
        from nn.losses import bce
        self.ratings.append((list(map(float, f)), 1 if rating > 0 else -1))
        X = np.array([r for r, _ in self.ratings])
        y = np.array([[1.0 if v > 0 else 0.0] for _, v in self.ratings])
        for _ in range(100):
            _, g = bce(self.net.forward(X), y)
            self.net.backward(g)
            self.opt.step()


class Stroke:
    """Рука с фломастером: ритм руки - три колебания (как кругами водить рукой), их силы по
    осям задаёт код рисунка z; фазы - свои у каждого человечка (его рука). Сначала рукой была
    случайная рекуррентная сеть - замкнутые рисунки у неё почти не получались, и красота не
    росла (замер в плане)."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed + 77)
        self.phase = rng.uniform(0, 2 * math.pi, (2, 3))
        self.lift = rng.uniform(0, 2 * math.pi)

    def draw(self, z):
        z = np.asarray(z, float)
        pos = np.zeros(2)
        pts, pens = [pos.copy()], []
        for t in range(STEPS):
            ph = 2 * math.pi * (t + 0.5) / STEPS
            vx = sum(z[k] * math.cos((k + 1) * ph + self.phase[0, k]) for k in range(3))
            vy = sum(z[3 + k] * math.sin((k + 1) * ph + self.phase[1, k]) for k in range(3))
            pos = pos + np.array([vx, vy]) * SCALE * 0.6
            pts.append(pos.copy())
            pens.append(0.0)
        return np.array(pts), np.array(pens)


class MarkerHeads:
    """Голова решает, что рисовать - код рисунка z (навык с гауссовым шумом, REINFORCE по
    красоте); рисует рука (Stroke). Сначала пробовал учить саму руку шаг за шагом - REINFORCE
    по 24 шагам разваливался в прямые линии (замер в плане)."""

    def __init__(self, seed=0):
        from skills.base import GaussianSkill
        self.code = GaussianSkill(1, [-2.0] * LATENT, [2.0] * LATENT, hidden=12, seed=seed, sigma=0.3, lr=1e-2)
        # новичок: мелкие быстрые колебания - каракуля
        unit = self.code.bounds.to_unit(np.array([0.1, 0.4, 1.9, 0.1, 0.4, 1.9], np.float32))
        self.code.net.items[-1].b[:] = np.arctanh(np.clip(unit, -0.95, 0.95))
        self.stroke = Stroke(seed)
        self.taste = DrawTaste(seed)
        self.explore = True

    @property
    def attempts(self):
        return self.code.attempts

    def draw(self):
        self.code.explore = self.explore
        z, rec = self.code.act(np.ones(1, np.float32))
        pts, pens = self.stroke.draw(np.asarray(z, float))
        return pts, pens, rec

    def learn(self, rec, reward):
        self.code.learn(rec, reward)

    def save(self, path):
        a = {"code." + k: v for k, v in io.collect(self.code.nets()).items()}
        a.update({"taste." + k: v for k, v in io.collect({"net": self.taste.net}).items()})
        a["meta"] = np.array([self.code.baseline, self.code.spread, self.code.attempts], float)
        a["ratings"] = np.array([r + [v] for r, v in self.taste.ratings] or [[0.0] * (FEATS + 1)])
        io.save(path, a)

    def load(self, path):
        a = io.load(path)
        io.restore(self.code.nets(), {k[5:]: v for k, v in a.items() if k.startswith("code.")})
        io.restore({"net": self.taste.net}, {k[6:]: v for k, v in a.items() if k.startswith("taste.")})
        b, s_, n = a["meta"]
        self.code.baseline, self.code.spread, self.code.attempts = float(b), float(s_), int(n)
        self.taste.ratings = [(list(r[:-1]), int(r[-1])) for r in a["ratings"] if r[-1] != 0]
        return self


class DrawAttempt:
    name = "marker"
    DRAW_TIME = 4.0

    def __init__(self, heads, ctx):
        self.heads, self.ctx = heads, ctx
        self.points, self.pens, self.recs = heads.draw()
        self.t = 0.0
        self.x0 = ctx.x + ctx.facing * 0.6
        self.y0 = ctx.ground(self.x0) + 0.9

    def step(self, dt, events):
        self.t += dt
        if int((self.t - dt) * 4) != int(self.t * 4):
            events.append(("sound", "scratch", 0.3, 0.05))
        if self.t < self.DRAW_TIME:
            return None
        f = features(self.points, self.pens)
        beauty = self.heads.taste.beauty(f)
        self.heads.learn(self.recs, beauty)
        drawing = {"x": round(self.x0, 2), "y": round(self.y0, 2), "facing": self.ctx.facing,
                   "points": [[round(float(px), 3), round(float(py), 3)] for px, py in self.points],
                   "pens": [int(v) for v in self.pens], "features": [round(float(v), 3) for v in f]}
        events.append(("drawing", drawing))
        return beauty, {"beauty": round(beauty, 3), "drawing": drawing}

    def finish_timeout(self):
        return 0.0, {"timeout": True}

    def pose(self, human):
        a = human.angles.copy()
        k = min(1.0, self.t / self.DRAW_TIME)
        n = min(len(self.points) - 1, int(k * (len(self.points) - 1)))
        dx, dy = self.points[n]
        a[3] = 1.4 + 2.0 * dy
        a[5] = 0.6 - 1.5 * dx
        return a

    def visuals(self):
        k = min(1.0, self.t / self.DRAW_TIME)
        n = int(k * (len(self.points) - 1))
        return [("drawing", self.x0, self.y0, (self.points[:n + 1], self.pens[:n], self.ctx.facing), (0.15, 0.2, 0.7))]


def practice(heads, rng=None):
    from toys.blocks import Flat
    from toys.inplace import Ctx, run
    return run(DrawAttempt(heads, Ctx(Flat(), 0.0)))[0]
