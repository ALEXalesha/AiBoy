"""Пузыри, самолётик, бумеранг - игрушки, которые летают (physics/aero.py) и которые несёт ветер
(world/weather.py). У каждой голова - навык с гауссовым шумом (REINFORCE): решает, как
дуть, как бросить. Новички начинают неумело (подобрано так, чтобы было чему учиться).

- **пузыри**: сила и долгота выдоха -> пузырь; дуть сильно - лопнет сразу; потом он ловит
  пузырь рукой, если тот пролетает рядом. Успех - размер пузыря, который не лопнул сам, и
  «лопнул рукой» - вдвое. Звук: выдох, хлопок (высота - по размеру);
- **самолётик**: угол, сила броска, балансировка -> дальность полёта (успех - до 40 м). Звук: взмах,
  шелест при посадке;
- **бумеранг**: угол, сила, вращение -> насколько близко вернулся; поймал (ближе POCKET) -
  вдвое. Звук: свист.
"""
import math

import numpy as np

from physics.aero import Boomerang, Bubble, Plane
from skills.base import GaussianSkill

HAND_H = 1.4             # м - высота руки над землёй
POCKET = 0.6             # м - бумеранг пойман


def _newbie(skill, values):
    unit = skill.bounds.to_unit(np.array(values, np.float32))
    skill.net.items[-1].b[:] = np.arctanh(np.clip(unit, -0.95, 0.95))


class Heads:
    """Одна голова на игрушку: вход - ветер (м/с, /4) и постоянная."""

    def __init__(self, lo, hi, newbie, seed=0, sigma=0.25):
        self.skill = GaussianSkill(2, lo, hi, hidden=16, seed=seed, sigma=sigma, lr=5e-3)
        _newbie(self.skill, newbie)

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


class BubbleHeads(Heads):
    def __init__(self, seed=0):
        # сила 0..1, долгота выдоха 0.2..2.5 с; новичок дует резко и коротко
        super().__init__([0.0, 0.2], [1.0, 2.5], [0.85, 0.4], seed)


class PlaneHeads(Heads):
    def __init__(self, seed=0):
        # угол броска, скорость, балансировка; новичок кидает вверх и с большой балансировкой
        super().__init__([-0.3, 2.0, -0.1], [1.0, 12.0, 0.8], [0.8, 9.0, 0.6], seed)
        self.skill = GaussianSkill(2, [-0.3, 2.0, -0.1], [1.0, 12.0, 0.8], hidden=16, seed=seed, sigma=0.25, lr=5e-3)
        _newbie(self.skill, [0.8, 9.0, 0.6])


class BoomerangHeads(Heads):
    def __init__(self, seed=0):
        # угол, скорость, вращение (рад/с); новичок - сильно и почти без вращения
        super().__init__([0.0, 3.0, 0.0], [1.0, 16.0, 120.0], [0.3, 14.0, 10.0], seed)
        self.skill = GaussianSkill(2, [0.0, 3.0, 0.0], [1.0, 16.0, 120.0], hidden=16, seed=seed, sigma=0.25, lr=5e-3)
        _newbie(self.skill, [0.3, 14.0, 10.0])


class FlyAttempt:
    """Общее для летающих: команда головы, полёт, итог."""
    name = "fly"
    SOUND = "whoosh"

    def __init__(self, heads, ctx):
        self.heads, self.ctx = heads, ctx
        self.rng = ctx.rng if ctx.rng is not None else np.random.default_rng()
        w = ctx.wind_at(ctx.x, ctx.ground(ctx.x) + 1.5, ctx.t)
        self.wind0 = w
        self.cmd, self.record = heads.skill.act(np.array([w / 4.0, 1.0], np.float32))
        self.cmd = [float(v) for v in self.cmd]
        self.t = 0.0
        self.trace = []
        self.started = False

    def wind(self, x, y):
        return self.ctx.wind_at(x, y, self.ctx.t + self.t)

    def done(self, success, reward, entry):
        self.heads.skill.learn(self.record, reward)
        entry["cmd"] = [round(v, 3) for v in self.cmd]
        entry["trace"] = self.trace[::4]
        return success, entry

    def finish_timeout(self):
        return self.done(0.0, 0.0, {"timeout": True})

    def pose(self, human):
        a = human.angles.copy()
        k = min(1.0, self.t / 0.3)
        a[3] = 2.4 - 1.4 * k if not self.started or self.t < 0.3 else 1.0
        return a


class BubbleAttempt(FlyAttempt):
    name = "bubbles"

    def __init__(self, heads, ctx):
        super().__init__(heads, ctx)
        self.strength, self.blow = self.cmd
        self.bubble = None
        self.popped_by_hand = False
        self.burst = False
        self.hand = (ctx.x + ctx.facing * 0.35, ctx.ground(ctx.x) + HAND_H)

    def step(self, dt, events):
        self.t += dt
        if self.bubble is None:
            if self.t >= self.blow:
                r = 0.015 + 0.06 * self.blow * (1.0 - 0.6 * self.strength) + 0.03 * self.strength
                self.bubble = Bubble(self.hand[0] + self.ctx.facing * 0.1, self.hand[1] + 0.1, r)
                events.append(("sound", "whoosh", 0.3 * self.strength + 0.1, r))
                # дул слишком сильно - лопнул сразу (чем больше пузырь, тем легче)
                p = 1.0 / (1.0 + math.exp(-((self.strength - 0.55) * 12 + (r - 0.06) * 30)))
                if self.rng.random() < p:
                    self.burst = True
                    events.append(("sound", "pop", 0.6, r))
                    # чем сильнее дул, тем хуже: иначе, лопаясь каждый раз одинаково, он не понял бы,
                    # в какую сторону меняться (замер: у одного зерна все 300 попыток лопнули)
                    return self.done(0.0, -0.5 - self.strength, {"size": round(r, 3), "burst": True})
            return None
        b = self.bubble
        b.step(self.wind(b.x, b.y), dt)
        self.trace.append((round(b.x - self.ctx.x, 2), round(b.y, 2)))
        # лопается сам: чем больше и чем дольше летит, тем вероятнее
        if self.rng.random() < (0.1 + 3.0 * b.r) * dt:
            events.append(("sound", "pop", 0.5, b.r))
            return self.done(min(1.0, b.r / 0.08) * 0.5, b.r / 0.08 * 0.5, {"size": round(b.r, 3), "burst": False})
        if math.hypot(b.x - self.hand[0], b.y - self.hand[1]) < 0.25 and b.age > 0.3:
            events.append(("sound", "pop", 0.8, b.r))
            return self.done(min(1.0, b.r / 0.08), b.r / 0.08, {"size": round(b.r, 3), "hand": True})
        if b.age > 4.0 or b.y > self.hand[1] + 6:
            return self.done(min(1.0, b.r / 0.08) * 0.5, b.r / 0.08 * 0.5, {"size": round(b.r, 3), "away": True})
        return None

    def visuals(self):
        if self.bubble is None:
            return []
        b = self.bubble
        return [("bubble", b.x, b.y, b.r, (0.8, 0.9, 1.0))]


class PlaneAttempt(FlyAttempt):
    name = "plane"

    def __init__(self, heads, ctx):
        super().__init__(heads, ctx)
        a, v, trim = self.cmd
        f = ctx.facing
        strength = ctx.arm / 0.55
        self.plane = Plane(ctx.x + f * 0.3, ctx.ground(ctx.x) + HAND_H + 0.2, f * v * strength * math.cos(a),
                           v * strength * math.sin(a), trim)
        if f < 0:
            self.plane.pitch = math.atan2(self.plane.vy, -self.plane.vx)   # в зеркальных координатах
        self.f = f
        self.stalls = 0

    def step(self, dt, events):
        if not self.started:
            self.started = True
            events.append(("sound", "whoosh", 0.4, 0.2))
        self.t += dt
        p = self.plane
        # самолётик летит в сторону взгляда; балансировка - в его собственных координатах
        alpha = p.step(self.wind(p.x, p.y), dt) if self.f > 0 else self._mirrored(p, dt)
        self.stalls += abs(alpha) > Plane.STALL
        self.trace.append((round(p.x - self.ctx.x, 2), round(p.y, 2)))
        if p.y <= self.ctx.ground(p.x) or p.time > 25:
            dist = abs(p.x - self.ctx.x)
            events.append(("sound", "land", 0.2, 0.1))
            success = min(1.0, dist / 40.0)
            return self.done(success, success, {"distance": round(dist, 2), "time": round(p.time, 2)})
        return None

    def _mirrored(self, p, dt):
        """Летит влево: считаем в зеркале (x -> -x), чтобы балансировка была своя, не перевёрнутая."""
        p.x, p.vx = -p.x, -p.vx
        alpha = p.step(-self.wind(-p.x, p.y), dt)
        p.x, p.vx = -p.x, -p.vx
        return alpha

    def visuals(self):
        p = self.plane
        return [("plane", p.x, p.y, (p.pitch if self.f > 0 else math.pi - p.pitch), (0.95, 0.95, 0.9))]


class BoomerangAttempt(FlyAttempt):
    name = "boomerang"

    def __init__(self, heads, ctx):
        super().__init__(heads, ctx)
        a, v, spin = self.cmd
        f = ctx.facing
        strength = ctx.arm / 0.55
        self.f = f
        self.b = Boomerang(0.0, ctx.ground(ctx.x) + HAND_H, v * strength * math.cos(a), v * strength * math.sin(a), spin)
        self.closest = None

    def pos(self):
        return self.ctx.x + self.f * self.b.x, self.b.y

    def step(self, dt, events):
        if not self.started:
            self.started = True
            events.append(("sound", "whoosh", 0.7, 0.3))
        self.t += dt
        b = self.b
        x, y = self.pos()
        b.step(self.f * self.wind(x, y), dt)
        x, y = self.pos()
        self.trace.append((round(x - self.ctx.x, 2), round(y, 2)))
        if b.time > 1.0:
            d = b.home()
            self.closest = d if self.closest is None else min(self.closest, d)
            if d < POCKET:
                events.append(("sound", "catch", 0.8, 0.2))
                return self.done(1.0, 2.0, {"return": round(d, 2), "caught": True})
        if y <= self.ctx.ground(x) or b.time > 10:
            d = self.closest if self.closest is not None else b.home()
            events.append(("sound", "land", 0.3, 0.2))
            success = math.exp(-d / 3.0)
            return self.done(success, success, {"return": round(d, 2), "caught": False})
        return None

    def visuals(self):
        x, y = self.pos()
        return [("boomerang", x, y, self.b.time * self.b.spin, (0.85, 0.55, 0.25))]


def practice(heads, attempt_cls, rng, wind=None, seed=0):
    from toys.blocks import Flat
    from toys.inplace import Ctx, run
    return run(attempt_cls(heads, Ctx(Flat(), 0.0, rng=rng, seed=seed, wind=wind)))[0]
