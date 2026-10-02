"""Удочка: закинуть, дождаться поклёвки, вовремя подсечь. Только где есть вода.

Две головы (навыки с гауссовым шумом, REINFORCE): одна решает угол и силу заброса и на какой
глубине держать наживку (доля глубины пруда), другая - через сколько подсекать после
поклёвки; она учится только на поклёвках (своя база), иначе её сигнал тонул в «клюнуло или
нет» (замер в плане). Поплавок
летит (бросок), падает в воду - если мимо воды, попытка пустая. Клюёт рыба по своим
правилам (world/fish.py: время суток и глубина); подсёк слишком рано или поздно - сорвалась.
Успех - поймал (1), иначе 0. Звук: заброс (свист), всплеск поплавка, катушка - всплеск.
Пойманная рыба - в «Галерею», в «Улов» (запись журнала с её видом).
"""
import math

import numpy as np

from skills.base import GaussianSkill

WAIT = 20.0              # с - ждёт поклёвки, потом сматывает
HOOK = (0.25, 0.9)       # с после поклёвки - вовремя (рыба держит наживку)
HAND_H = 1.4


class RodHeads:
    def __init__(self, seed=0):
        # угол, сила (м/с), глубина наживки (доля); новичок - бросает слабо, наживку у поверхности
        self.skill = GaussianSkill(2, [0.1, 2.0, 0.0], [1.2, 12.0, 1.0], hidden=16, seed=seed, sigma=0.25, lr=5e-3)
        unit = self.skill.bounds.to_unit(np.array([0.9, 3.0, 0.05], np.float32))
        self.skill.net.items[-1].b[:] = np.arctanh(np.clip(unit, -0.95, 0.95))
        # когда подсекать после поклёвки (с); новичок - сразу
        self.hook = GaussianSkill(1, [0.0], [2.0], hidden=8, seed=seed + 1, sigma=0.3, lr=1e-2)
        self.hook.net.items[-1].b[:] = np.arctanh(self.hook.bounds.to_unit(np.array([0.05], np.float32)))

    def save(self, path):
        from nn import io
        a = {"cast." + k: v for k, v in io.collect(self.skill.nets()).items()}
        a.update({"hook." + k: v for k, v in io.collect(self.hook.nets()).items()})
        a["meta"] = np.array([self.skill.baseline, self.skill.spread, self.skill.attempts,
                              self.hook.baseline, self.hook.spread, self.hook.attempts], float)
        io.save(path, a)

    def load(self, path):
        from nn import io
        a = io.load(path)
        io.restore(self.skill.nets(), {k[5:]: v for k, v in a.items() if k.startswith("cast.")})
        io.restore(self.hook.nets(), {k[5:]: v for k, v in a.items() if k.startswith("hook.")})
        m = a["meta"]
        self.skill.baseline, self.skill.spread, self.skill.attempts = float(m[0]), float(m[1]), int(m[2])
        self.hook.baseline, self.hook.spread, self.hook.attempts = float(m[3]), float(m[4]), int(m[5])
        return self


class RodAttempt:
    name = "rod"

    def __init__(self, heads, ctx):
        self.heads, self.ctx = heads, ctx
        self.rng = ctx.rng if ctx.rng is not None else np.random.default_rng()
        water = ctx.water
        # к какому пруду ближе всего (он стоит у берега: в жизни сессия начинается у воды)
        d, wpos = 1e9, 0.0
        if water is not None:
            for p in water.ponds:
                mid = (p.left + ((p.right - p.left) % ctx.world.width) / 2) % ctx.world.width
                dist = abs((mid - ctx.x + ctx.world.width / 2) % ctx.world.width - ctx.world.width / 2)
                if dist < d:
                    d, wpos = dist, mid
        self.facing = 1 if ((wpos - ctx.x + ctx.world.width / 2) % ctx.world.width - ctx.world.width / 2) >= 0 else -1
        self.cmd, self.record = heads.skill.act(np.array([min(d, 15.0) / 15.0, 1.0], np.float32))
        self.angle, self.speed, self.depth = [float(v) for v in self.cmd]
        hook, self.hook_record = heads.hook.act(np.ones(1, np.float32))
        self.hook = float(hook[0])
        self.cmd = list(self.cmd) + [self.hook]
        self.stage = "cast"
        self.t = 0.0
        self.x, self.y = ctx.x + self.facing * 0.3, ctx.ground(ctx.x) + HAND_H
        self.vx, self.vy = self.facing * self.speed * math.cos(self.angle), self.speed * math.sin(self.angle)
        self.bite_at = None
        self.fish = None
        self.trace = []

    def step(self, dt, events):
        self.t += dt
        ctx = self.ctx
        if self.stage == "cast":
            if self.t <= dt:
                events.append(("sound", "whoosh", 0.5, 0.1))
            self.vy -= 9.81 * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.trace.append((round(self.x - ctx.x, 2), round(self.y, 2)))
            level = ctx.water.level_at(self.x) if ctx.water is not None else None
            if level is not None and self.y <= level:
                events.append(("sound", "splash", 0.4, 0.05))
                self.stage, self.t, self.y = "wait", 0.0, level
                self.pond = ctx.water.pond_at(self.x)
            elif self.y <= ctx.ground(self.x):
                return self.done(False, "мимо воды")
            return None
        if self.stage == "wait":
            if ctx.fish is not None:
                idx = ctx.water.ponds.index(self.pond) if self.pond in ctx.water.ponds else -1
                near = [f for f in ctx.fish.fish if f.pond == idx]
                for f in near:
                    if self.rng.random() < ctx.fish.bite_chance(ctx.t + self.t, self.depth, f, dt):
                        self.fish, self.stage, self.t = f, "bite", 0.0
                        events.append(("sound", "splash", 0.2, 0.03))
                        return None
            if self.t > WAIT:
                return self.done(False, "не клюёт")
            return None
        if self.stage == "bite":
            if self.t >= self.hook:
                ok = HOOK[0] <= self.hook <= HOOK[1]
                events.append(("sound", "splash", 0.7 if ok else 0.3, 0.1))
                return self.done(ok, "поймал" if ok else ("рано" if self.hook < HOOK[0] else "поздно"))
        return None

    def done(self, caught, why):
        reward = 1.0 if caught else (-0.5 if why == "мимо воды" else 0.0)
        if why in ("рано", "поздно"):
            # была поклёвка (наживка на верной глубине), и он чувствует, насколько промахнулся
            # с подсечкой: рыба дёрнула раньше или ещё держала. Без этого подсечка не учится:
            # одно «поймал / не поймал» почти не встречалось (замер в плане)
            gap = HOOK[0] - self.hook if why == "рано" else self.hook - HOOK[1]
            reward = 0.3 + 0.5 * math.exp(-max(0.0, gap) / 0.3)
        self.heads.skill.learn(self.record, reward)
        if why in ("поймал", "рано", "поздно"):
            self.heads.hook.learn(self.hook_record, 1.0 if caught else reward - 0.3)
        entry = {"caught": bool(caught), "why": why, "cmd": [round(v, 3) for v in self.cmd], "trace": self.trace[::3]}
        if caught and self.fish is not None:
            f = self.fish
            entry["fish"] = {"length": round(f.length, 3), "height": round(f.height, 3), "tail": round(f.tail, 3),
                             "color": list(f.color)}
        return (1.0 if caught else 0.0), entry

    def finish_timeout(self):
        return self.done(False, "долго")

    def pose(self, human):
        a = human.angles.copy()
        if self.stage == "cast":
            a[3] = 2.6 - 1.6 * min(1.0, self.t / 0.3)
        else:
            a[3] = 1.3 if self.stage == "wait" else 2.2
        a[5] = 0.4
        return a

    def visuals(self):
        return [("bobber", self.x, self.y, 0.04, (0.95, 0.3, 0.2)),
                ("line", self.ctx.x + self.facing * 0.5, self.ctx.ground(self.ctx.x) + HAND_H + 0.6, self.x, self.y)]


def practice(heads, rng, world, water, fish, x, t0=0.0):
    from toys.inplace import Ctx, run
    return run(RodAttempt(heads, Ctx(world, x, rng=rng, water=water, fish=fish, t=t0)), limit=40.0)[0]


from toys.inplace import InPlaceSession  # noqa: E402


class RodSession(InPlaceSession):
    """Удочка в жизни: сначала - к берегу водоёма, который он видел (ближайший), потом попытки."""

    SHORE = 0.8              # м от воды

    def begin(self):
        self.active = True
        life = self.life
        spot = self.shore()
        if spot is None:
            self.active = False
            return
        self.going = spot
        life.walk.goal = spot
        self.pose = None

    def shore(self):
        life = self.life
        best = None
        for p in life.water.ponds:
            for edge, inward in ((p.left, 1), (p.right, -1)):
                spot = (edge - inward * self.SHORE) % life.world.width
                d = life.inv.dist(life.human.px, spot)
                if best is None or d < best[1]:
                    best = (spot, d)
        return None if best is None else best[0]

    def tick(self, dt, events):
        if not self.active:
            return False
        if getattr(self, "going", None) is not None:
            life = self.life
            if life.inv.dist(life.human.px, self.going) < 0.6:
                self.going = None
                life.walk.goal = None
                self.start_attempt()
            return True
        return super().tick(dt, events)
