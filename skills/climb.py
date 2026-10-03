"""Навык «лезть по стене» (спецификация 2.0, раздел 8a) - с нуля, как плавание.

Стена горы - зацепы снизу доверху (через HOLD_STEP м), у каждого своя надёжность (сеть
зацепов по зерну мира и горы) и свой наклон грани (для ледоруба). Каждый ход руки ищут зацеп,
ноги толкают. Навык (skills/base.GaussianSkill, REINFORCE) по тому, какие зацепы над ним (на
четверть, половину, три четверти и целую длину руки выше), как устали руки, на какой он высоте и
есть ли верёвка, решает:
- **докуда тянуться** (в длинах руки) - к какому зацепу;
- **как крепко держаться** (крепче - надёжнее на плохом зацепе, но руки устают быстрее);
- **как сильно толкнуться ногами** (дальше достаёт, но тоже усталость).
Вышло - выше на расстояние до зацепа; не вышло - срыв: без верёвки - съезжает к подножию
(безопасно, «ой»), с верёвкой - до крюка. Награда хода - насколько поднялся, срыв - минус.
Новичок тянется наугад и срывается; с попытками - выше (кривая в плане).
"""
import math

import numpy as np

from skills.base import GaussianSkill
from world.cppn import ACTS, Z_DIM, seed_vector

HOLD_STEP = 0.35            # м между зацепами
REACH = (0.2, 1.3)          # длин руки
FAIL_REWARD = -0.5
FATIGUE_MOVE = 0.006        # усталость рук за ход
FATIGUE_PUSH = 0.015        # и за толчок ногами
AXE_FATIGUE = 0.004         # и за удар ледорубом
HOOK_Q = 0.85               # крюк цепляется только за надёжный зацеп
ROPE_EVERY = 4.0            # м - перебрасывает крюк выше
MAX_MOVES = 600
LEDGE = 12.0                # м - на стене полки: на них стоит и отдыхает, срыв дальше полки не уносит
FALLS = 4                   # срывов за попытку - и хватит
HOLD_LAYERS = (Z_DIM + 3, 10, 2)


def _hold_net():
    rng = np.random.default_rng(0xC11)
    ws, bs = [], []
    for a, b in zip(HOLD_LAYERS, HOLD_LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.4 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_HOLDS = _hold_net()


def holds(world_seed, index, height):
    """Зацепы стены: (высоты от подножия, надёжность 0..1, наклон грани -1..1)."""
    ys = np.arange(HOLD_STEP, height + HOLD_STEP, HOLD_STEP)
    z = seed_vector(int(world_seed) * 7 + 11 * int(index) + 1)
    u = ys / max(height, 1.0)
    x = np.concatenate([np.broadcast_to(z, (len(ys), Z_DIM)), np.c_[np.sin(7.0 * u * math.pi), np.cos(3.0 * u * math.pi),
                                                                   np.sin(ys * 1.7)]], axis=1)
    ws, bs = _HOLDS
    h = x
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    q = 0.6 + 0.4 / (1.0 + np.exp(-1.8 * h[:, 0]))      # 0.6..1: есть за что взяться, но не везде надёжно
    return ys, q, np.tanh(h[:, 1])


class Wall:
    """Одна стена: зацепы и их надёжность (ледоруб её поднимает на время подъёма)."""

    def __init__(self, world_seed, index, height):
        self.height = float(height)
        self.ys, self.q, self.facet = holds(world_seed, index, height)
        self.boost = np.zeros(len(self.ys))
        self.ledge = (np.abs((self.ys + 1e-6) % LEDGE) < HOLD_STEP * 0.6) & (self.ys > 1.0)
        self.q[self.ledge] = 1.0

    def ledge_below(self, y):
        """Самая высокая полка не выше y (0 - подножие)."""
        below = self.ys[self.ledge & (self.ys <= y + 1e-6)]
        return float(below.max()) if len(below) else 0.0

    def quality(self, i):
        return float(min(1.0, self.q[i] + self.boost[i]))

    def above(self, y):
        """Индекс первого зацепа выше y."""
        return int(np.clip(np.searchsorted(self.ys, y + 0.05), 0, len(self.ys) - 1))

    def near(self, y):
        """Индекс зацепа ближе всего к высоте y."""
        return int(np.clip(round(y / HOLD_STEP) - 1, 0, len(self.ys) - 1))


class Climber:
    def __init__(self, seed=0):
        # вход: три следующих зацепа (надёжность, грань, сколько до него в руках), усталость, доля
        # высоты, верёвка; выход: к какому из трёх, угол кисти (под грань), толчок ногами
        self.skill = GaussianSkill(12, [0.0, -1.0, 0.0], [1.0, 1.0, 1.0], hidden=24, seed=seed, sigma=0.3, lr=3e-3)
        self.attempts = 0
        self.history = []                    # доля высоты за попытку
        self.tops = 0

    def features(self, wall, y, arm, fatigue, rope):
        i = wall.above(y)
        f = []
        for k in range(3):
            j = min(len(wall.ys) - 1, i + k)
            f += [wall.quality(j), float(wall.facet[j]), float(wall.ys[j] - y) / arm]
        return np.array(f + [fatigue, y / wall.height, 1.0 if rope else 0.0], np.float32)

    def confidence(self):
        """Как ему кажется, получится ли: средняя доля высоты последних 10 попыток."""
        return float(np.mean(self.history[-10:])) if self.history else 0.25

    def save(self, path):
        from nn import io
        arrays = io.collect(self.skill.nets())
        arrays["meta.numbers"] = np.array([self.skill.baseline, self.skill.spread, self.skill.attempts,
                                           self.attempts, self.tops], np.float64)
        arrays["meta.history"] = np.array(self.history[-50:] or [0.0], np.float64)
        io.save(path, arrays)

    def load(self, path):
        from nn import io
        a = io.load(path)
        io.restore(self.skill.nets(), a)
        b, s, n, m, t = a["meta.numbers"]
        self.skill.baseline, self.skill.spread, self.skill.attempts = float(b), float(s), int(n)
        self.attempts, self.tops = int(m), int(t)
        self.history = [float(v) for v in a["meta.history"]] if self.attempts else []
        return self


def move(climber, wall, y, arm, fatigue, rope, rng, axe=None, learn=True):
    """Один ход. Возвращает (новая высота или None при срыве, усталость, индекс зацепа).

    Успех - надёжность зацепа, угол кисти под его грань (главное, чему учится навык: новичок
    берётся как попало), не дальше, чем достаёт рука с толчком ног, и усталость."""
    feats = climber.features(wall, y, arm, fatigue, rope)
    cmd, rec = climber.skill.act(feats)
    pick, angle, push = float(cmd[0]), float(cmd[1]), float(cmd[2])
    k = min(2, int(pick * 3.0))
    i = min(len(wall.ys) - 1, wall.above(y) + k)
    gain = float(wall.ys[i] - y)
    q = wall.quality(i)
    over = max(0.0, gain - arm * (0.6 + 0.9 * push)) / arm
    grip = math.exp(-3.0 * (angle - float(wall.facet[i])) ** 2)
    if axe is not None:
        # ледоруб вбит в этот зацеп: хорошо вбил - держит как ступенька и подтянуться выше
        held = axe.strike(wall, i, rng, learn=learn)
        q = max(q, held)
        grip = max(grip, held)
        over = max(0.0, over - 0.3 * held)
        fatigue = min(1.0, fatigue + AXE_FATIGUE)
    # умелому ход удаётся в 96-98 % (иначе на полку в 12 м - 20-35 ходов - не пройти ни разу:
    # замер - навык упирался в 5 м стены 24 м)
    p = q ** 0.15 * grip * math.exp(-5.0 * over) * (1.0 - 0.25 * fatigue ** 2) * (1.03 if rope else 1.0)
    ok = rng.random() < min(0.99, p)
    fatigue = min(1.0, fatigue + FATIGUE_MOVE + FATIGUE_PUSH * push)
    if learn:
        climber.skill.learn(rec, gain / arm if ok else FAIL_REWARD)
    return (float(wall.ys[i]) if ok else None), fatigue, i


def practice(climber, wall, rng, arm=0.55, rope_every=None, rope_skill=None, axe=None, learn=True):
    """Одна попытка без жизни вокруг (законы, замеры): от подножия, пока не вершина или FALLS
    срывов (срыв - до последней полки, с верёвкой - до крюка, если он выше). Возвращает долю
    высоты (1 - залез)."""
    y, fatigue, best, falls, anchor = 0.0, 0.0, 0.0, 0, None
    next_rope = 0.0
    for _ in range(MAX_MOVES):
        if rope_every is not None and y >= next_rope and rope_skill is not None:
            hooked = rope_skill.throw(wall, y, rng, learn=learn)
            if hooked is not None:
                anchor = hooked
            next_rope = y + rope_every
        ny, fatigue, _ = move(climber, wall, y, arm, fatigue, anchor is not None and anchor > y, rng, axe, learn)
        if ny is None:
            falls += 1
            if falls >= FALLS:
                break
            stop = wall.ledge_below(y)
            if anchor is not None and anchor <= y:
                stop = max(stop, anchor)
            y = stop
            fatigue = 0.0 if stop in wall.ys[wall.ledge] or stop == 0.0 else max(0.0, fatigue - 0.3)
            continue
        if wall.ledge[wall.near(ny)]:
            fatigue = 0.0                        # на полке - стоит, руки отдыхают
        y = ny
        best = max(best, y)
        if y >= wall.height - HOLD_STEP / 2:
            best = wall.height
            break
    share = min(1.0, best / wall.height)
    climber.attempts += 1
    climber.history = (climber.history + [share])[-50:]
    if share >= 1.0:
        climber.tops += 1
    return share


class RopeSkill:
    """Верёвка с крюком: голова решает, насколько выше бросить крюк (1.5-6 м), по надёжности
    зацепов на 2, 3, 4, 5 и 6 м выше. Чем выше бросок, тем больше разброс; крюк цепляется за
    ближайший к месту падения зацеп, только если он надёжный (HOOK_Q). Зацепился - срыв
    останавливается на крюке. Награда - высота крюка, промах - минус."""

    def __init__(self, seed=0):
        self.skill = GaussianSkill(5, [1.5], [6.0], hidden=12, seed=seed, sigma=0.3, lr=5e-3)
        self.throws = self.hooked = 0
        self.history = []

    def throw(self, wall, y, rng, learn=True):
        feats = np.array([wall.quality(wall.near(y + d)) for d in (2.0, 3.0, 4.0, 5.0, 6.0)], np.float32)
        cmd, rec = self.skill.act(feats)
        aim = float(cmd[0])
        land = y + aim + rng.normal(0.0, 0.1 + 0.08 * aim)
        if land >= wall.height:
            land = wall.height - HOLD_STEP
        i = wall.near(land)
        ok = wall.quality(i) >= HOOK_Q and wall.ys[i] > y
        self.throws += 1
        self.hooked += int(ok)
        if learn:
            self.skill.learn(rec, (float(wall.ys[i] - y) / 6.0) if ok else -0.3)
        return float(wall.ys[i]) if ok else None

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


class AxeSkill:
    """Ледоруб: голова решает угол удара по грани и надёжности зацепа. Попал под грань -
    зацеп держит (1), мимо - как без ледоруба (0). Награда - насколько держит."""

    def __init__(self, seed=0):
        self.skill = GaussianSkill(2, [-1.0, 0.2], [1.0, 1.0], hidden=12, seed=seed, sigma=0.3, lr=5e-3)
        self.history = []

    def strike(self, wall, i, rng, learn=True):
        feats = np.array([float(wall.facet[i]), wall.quality(i)], np.float32)
        cmd, rec = self.skill.act(feats)
        angle, force = float(cmd[0]), float(cmd[1])
        held = math.exp(-4.0 * (angle - float(wall.facet[i])) ** 2) * min(1.0, 0.4 + force)
        if learn:
            self.skill.learn(rec, held - 0.2 * force)
        return held

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self
