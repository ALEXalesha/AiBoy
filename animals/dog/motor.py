"""Сеть движения собаки - голова походки (спецификация 2.0, раздел 8): раз в BOUT секунд по
тому, что впереди (уклон), как быстро она хочет (шаг, бег), в воде ли и устала ли, решает
частоту шага, размах, подъём лапы и долю опоры. Учится REINFORCE (skills/base.GaussianSkill) по
своей моторной награде: насколько быстро продвинулась туда, куда хотела, к желаемой скорости,
минус крен (кувырок - сильно), минус усилие.

В новом мире собака неуклюжая: сеть с нуля, походка случайная около середины - то семенит,
то кувыркается; через несколько минут бегает ровно (замер в плане).
"""
import math

import numpy as np

from skills.base import GaussianSkill

BOUT = 3.0                  # с - одно решение походки
SETTLE = 1.0                # с - первая секунда - переход от прежней походки, в награду не идёт
LO = [1.0, 0.1, 0.05, 0.3]  # частота (Гц), размах (рад), подъём лапы (доля ноги), доля опоры
HI = [4.0, 0.6, 0.5, 0.8]
WALK, RUN = 1.0, 2.5        # м/с - желаемая скорость шагом и бегом
TILT = 0.45                 # рад - крен больше - штраф
NEWBIE = 0.9                # разброс привычки новичка (в долях -1..1 до tanh)
GENTLE = 0.3                # уклон впереди круче - забег не учит (замер в плане)


class DogMotor:
    def __init__(self, seed=0):
        self.skill = GaussianSkill(4, LO, HI, hidden=16, seed=seed, sigma=0.25, lr=2e-3)
        # новичок - со своей неуклюжей привычкой (семенит, запинается, прыгает козлом): сдвиг
        # выхода сети по зерну; середина диапазона уже бегала почти как лучшая походка
        # (замер в плане), и учиться было нечему
        last = self.skill.net.items[-1]
        last.b[:] = np.random.default_rng(seed + 11).normal(0.0, NEWBIE, last.b.shape).astype(np.float32)
        self.bouts = 0
        self.slope_base = np.zeros(6)            # обычная награда по уклону впереди

    def features(self, slope, want, wet, tired):
        return np.array([float(np.clip(slope, -1.5, 1.5)), want / RUN, float(wet), float(tired)], np.float32)

    def act(self, feats):
        cmd, rec = self.skill.act(feats)
        return tuple(float(c) for c in cmd), rec

    @staticmethod
    def reward(progress, want, worst_tilt, cmd, wet=0.0):
        """progress - сколько м продвинулась в желаемую сторону за BOUT - SETTLE."""
        if want <= 0:
            return 0.0
        speed = progress / (BOUT - SETTLE)
        r = float(np.clip(speed / want, -1.0, 1.3))
        r -= 1.5 * max(0.0, worst_tilt - TILT)
        r -= 0.05 * cmd[0] * cmd[1]                          # усилие
        # кувырок - плохо, но не в разы хуже, чем стоять на месте: штрафы -2..-3 раскачивали
        # REINFORCE, и выученная походка портилась дальше (замер: 0.54 -> -0.74 за 300 забегов)
        return max(-1.0, r)

    def learn(self, rec, reward):
        """Награда - против обычной для такого уклона: в гору любая походка медленнее, и без
        своей базы на уклон разброс склонов глушил разницу походок (замер: на холмах сеть не
        училась, на ровном - училась)."""
        self.bouts += 1
        x = rec[0]
        if abs(float(x[0])) > GENTLE:
            return 0.0                           # на крутом награда - от склона, не от походки
        i = int(np.clip((float(x[0]) + 1.5) / 3.0 * len(self.slope_base), 0, len(self.slope_base) - 1))
        adj = reward - self.slope_base[i]
        self.slope_base[i] += 0.05 * adj
        return self.skill.learn(rec, adj)

    def save(self, path):
        from nn import io
        arrays = io.collect(self.skill.nets())
        arrays["meta.numbers"] = np.array([self.skill.baseline, self.skill.spread, self.skill.attempts], np.float64)
        arrays["meta.slope"] = self.slope_base.astype(np.float64)
        io.save(path, arrays)

    def load(self, path):
        from nn import io
        arrays = io.load(path)
        io.restore(self.skill.nets(), arrays)
        b, s, n = arrays["meta.numbers"]
        self.skill.baseline, self.skill.spread, self.skill.attempts = float(b), float(s), int(n)
        if "meta.slope" in arrays and arrays["meta.slope"].shape == self.slope_base.shape:
            self.slope_base = arrays["meta.slope"].astype(np.float64)
        self.bouts = self.skill.attempts
        return self


def tilt(world, d):
    """Крен собаки относительно земли под ней (на склоне спина идёт вдоль склона - это не крен)."""
    g = math.atan2(world.height_at(d.x + d.L / 2) - world.height_at(d.x - d.L / 2), d.L)
    return abs((d.angle() - g + math.pi) % (2 * math.pi) - math.pi)


def slope_ahead(world, x, facing, dist=1.0):
    return (world.height_at(x + facing * dist) - world.height_at(x)) / dist


def practice(motor, world, x0, rng=None, seconds=9.0, look_=None, want=RUN, learn=True):
    """Один забег без жизни вокруг (замеры, тесты): собака бежит seconds секунд по миру world от
    x0 в случайную сторону, сеть решает походку раз в BOUT и учится. Возвращает среднюю
    моторную награду забега (скорость к желаемой минус крен и усилие)."""
    from animals.dog.body import DogBody
    from animals.dog.look import look
    from physics.world import PhysWorld
    rng = rng or np.random.default_rng()
    p = PhysWorld(world, x0)
    facing = 1 if rng.random() < 0.5 else -1
    d = DogBody(p, x0, look_ or look(0), facing)
    t = 0.0
    steps = int(BOUT * 60)
    rewards = []
    while t < seconds:
        feats = motor.features(slope_ahead(world, d.x, facing), want, 0.0, 0.0)
        cmd, rec = motor.act(feats)
        d.gait = cmd
        x_before, worst = None, 0.0
        for k in range(steps):
            if k == int(SETTLE * 60):
                x_before = d.x
            p.follow(d.x)
            d.step(1 / 60)
            p.step(1 / 60)
            if x_before is not None:
                worst = max(worst, tilt(world, d))
        t += BOUT
        r = DogMotor.reward(facing * (d.x - x_before), want, worst, cmd)
        rewards.append(r)
        if learn:
            motor.learn(rec, r)
    return float(np.mean(rewards))
