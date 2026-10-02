"""Кубики: строит башню. Какие кубики - размеры и цвета - из сети игрушек по зерну сундука.

Попытка - одна башня до TOWER кубиков, кубики разные. Голова решает, какой кубик взять
следующим (сеть выбора: широкий на узкий - упадёт) и как поставить: сдвиг относительно
верхнего и наклон (навык с гауссовым шумом, REINFORCE); рука ставит не идеально (шум руки).
Новичок ставит небрежно - с наклоном (так начинается и учится). Поставил - ждёт SETTLE с: стоит - следующий, упала - конец. Успех - высота
выстоявшей SETTLE с башни относительно наибольшей. Физика - pymunk (physics/world.py):
неуравновешенная башня падает, ровная стоит (закон).
Звук: щелчок кубика о кубик, глухой удар при падении.
"""
import math

import numpy as np

from physics.world import PhysWorld
from acts.policy import ChoicePolicy
from skills.base import GaussianSkill
from toys.inplace import run

TOWER = 6
SETTLE = 1.0             # с ждёт после каждого кубика
STAND = 3.0              # с - в конце башня должна простоять
HAND = 0.025             # м - шум руки
CLUMSY = 0.22            # рад - наклон, с которым ставит новичок
LEAN = 0.05              # м - и сдвиг вбок (без привычки голове укладки нечему было учиться:
                         # середина и так лучшая, замер мутацией)
FALL = 0.06              # м - сдвинулся больше - упало


def sizes(seed):
    """Размеры кубиков (ширина, высота) из сети игрушек."""
    from toys.base import look
    out = []
    for k in range(TOWER):
        _, _, prop = look(seed, 10 + k)
        w = 0.07 + 0.03 * k + 0.08 * prop          # от узких к широким, у каждого сундука по-своему
        out.append((round(w, 3), round(0.08 + 0.05 * (1 - prop), 3)))
    return out


class BlockHeads:
    def __init__(self, seed=0):
        # вход: какой по счёту кубик (доля), ширина верхнего и этого (м)
        self.place = GaussianSkill(3, [-0.12, -0.3], [0.12, 0.3], hidden=16, seed=seed, sigma=0.3, lr=5e-3)
        self.place.net.items[-1].b[:] = [np.arctanh(LEAN / 0.12), np.arctanh(CLUMSY / 0.3)]
        self.pick = ChoicePolicy(2 + TOWER, TOWER, hidden=16, seed=seed + 1, lr=5e-3, entropy=0.05)
        self.attempts = 0

    def save(self, path):
        from nn import io
        arrays = {"place." + k: v for k, v in io.collect(self.place.nets()).items()}
        arrays.update({"pick." + k: v for k, v in io.collect({"net": self.pick.net}).items()})
        arrays["meta"] = np.array([self.place.baseline, self.place.spread, self.place.attempts, self.attempts], float)
        io.save(path, arrays)

    def load(self, path):
        from nn import io
        a = io.load(path)
        io.restore(self.place.nets(), {k[6:]: v for k, v in a.items() if k.startswith("place.")})
        io.restore({"net": self.pick.net}, {k[5:]: v for k, v in a.items() if k.startswith("pick.")})
        b, s, n, m = a["meta"]
        self.place.baseline, self.place.spread, self.place.attempts, self.attempts = float(b), float(s), int(n), int(m)
        return self


class Flat:
    width, dx = 200.0, 0.25

    def height_at(self, x):
        return 0.0


class TowerAttempt:
    def __init__(self, heads, ctx):
        self.heads, self.ctx = heads, ctx
        self.rng = ctx.rng if ctx.rng is not None else np.random.default_rng()
        self.sizes = sizes(ctx.seed)
        self.x0 = ctx.x + ctx.facing * 0.6
        self.phys = PhysWorld(ctx.world, center=self.x0)
        self.placed = []             # (имя, x, y цели)
        self.order = []              # какие кубики по порядку
        self.records = []
        self.picks = []
        self.t = 0.0
        self.stage = "place"
        self.height = 0.0

    def top(self):
        if not self.placed:
            return self.x0, self.ctx.ground(self.x0), 0.5
        name, _, _ = self.placed[-1]
        b = self.phys.body(name)
        w, h = self.sizes[self.order[-1]]
        return float(b.position.x), float(b.position.y) + h / 2, w

    def place(self, events):
        k = len(self.placed)
        tx, ty, tw = self.top()
        left = np.array([i not in self.order for i in range(TOWER)])
        feats = np.concatenate([[k / TOWER, tw], [self.sizes[i][0] if left[i] else 0.0 for i in range(TOWER)]])
        choice, _ = self.heads.pick.choose(feats, left, rng=self.rng)
        self.picks.append((feats, left, choice))
        self.order.append(choice)
        w, h = self.sizes[choice]
        cmd, rec = self.heads.place.act(np.array([k / TOWER, tw, w], np.float32))
        dx, ang = float(cmd[0]), float(cmd[1])
        x = tx + dx + self.rng.normal(0.0, HAND)
        y = ty + h / 2 + 0.005
        name = f"кубик{k}"
        self.phys.add_box(name, x, y, w, h, mass=0.3, angle=ang)
        self.placed.append((name, x, y))
        self.records.append(rec)
        events.append(("sound", "clack", 0.6, w))
        self.t = 0.0

    def fallen(self):
        for i, (name, x, y) in enumerate(self.placed):
            b = self.phys.body(name)
            if b is None or math.hypot(b.position.x - x, b.position.y - y) > FALL:
                return True
        return False

    def step(self, dt, events):
        if self.stage == "place":
            self.place(events)
            self.stage = "settle"
            return None
        self.phys.step(dt)
        self.t += dt
        if self.fallen():
            events.append(("sound", "fall", 0.8, 0.2))
            return self.finish(len(self.placed) - 1)
        if self.stage == "settle" and self.t >= SETTLE:
            if len(self.placed) >= TOWER:
                self.stage = "stand"
                self.t = 0.0
            else:
                self.stage = "place"
        elif self.stage == "stand" and self.t >= STAND:
            return self.finish(len(self.placed))
        return None

    def finish(self, standing):
        standing = max(0, standing)
        height = sum(self.sizes[self.order[i]][1] for i in range(standing))
        best = sum(h for _, h in self.sizes)
        success = height / best
        # все решения башни учатся на её итоге (высота): «выстоял ли этот кубик» поощрял класть
        # узкие вниз - потом на них падали широкие (замер: успех падал с 0.67 до 0.33)
        self.heads.place.learn_many(self.records, success)
        for feats, left, choice in self.picks:
            self.heads.pick.learn(feats, left, choice, success)
        self.heads.attempts += 1
        self.height = height
        return success, {"blocks": standing, "height": round(height, 3)}

    def finish_timeout(self):
        return self.finish(len(self.placed))

    def pose(self, human):
        a = human.angles.copy()
        k = min(1.0, self.t / 0.4) if self.stage == "settle" else 0.0
        a[0] = 0.4                               # наклонился к башне
        a[3] = 1.2 - 0.4 * k
        a[5] = 0.5
        return a

    def visuals(self):
        out = []
        from toys.base import look
        for i, (name, _, _) in enumerate(self.placed):
            b = self.phys.body(name)
            if b is None:
                continue
            w, h = self.sizes[self.order[i]]
            c1, _, _ = look(self.ctx.seed, 10 + self.order[i])
            out.append(("box", float(b.position.x), float(b.position.y), (w, h, float(b.angle)), c1))
        return out


def practice(heads, rng, seed=0):
    from toys.inplace import Ctx
    return run(TowerAttempt(heads, Ctx(Flat(), 0.0, rng=rng, seed=seed)))[0]
