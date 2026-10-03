"""Полёт птицы (спецификация 2.0, раздел 9): точка с крыльями и своей аэродинамикой.

Подъёмная сила растёт с квадратом скорости и с «хвостом» (угол атаки), сопротивление - тоже;
взмахи дают тягу в сторону «наклона взмаха». Парители - большая площадь крыла и слабый взмах,
мелкие - наоборот. Сеть полёта (skills/base.GaussianSkill) раз в DECIDE с по тому, насколько он
выше или ниже нужной высоты, быстрее или медленнее нужного, как падает и какой ветер, решает силу
взмахов, их наклон и хвост. Награда - держать высоту и скорость, мало тратя сил. Первые птицы
нового мира машут неуклюже (привычка новичка по зерну) и теряют высоту; потом летают ровно.
"""
import math

import numpy as np

from skills.base import GaussianSkill

G = 9.81
DECIDE = 0.5
FLIGHT = 12.0               # с - пробный полёт
NEWBIE = 0.5               # с 0.9 половина новичков сразу садилась на землю, а там награда одна
                           # и та же что ни делай - учиться не по чему (замер)
CRASH = -2.0
VDAMP = 1.5                 # 1/с


class Wings:
    """Аэродинамика породы: подъём крыла на крейсерской скорости, вертикальная и горизонтальная тяга
    взмаха, сопротивление на крейсерской, крейсерская скорость. Ровный полёт - взмах ~0.5 (мелкие)
    или ~0.3 (парители) и хвост посередине."""

    def __init__(self, look):
        if look.glider:
            self.lift, self.up, self.push, self.drag, self.cruise = 8.5, 4.0, 3.0, 1.0, 8.0
        else:
            self.lift, self.up, self.push, self.drag, self.cruise = 6.0, 8.0, 5.0, 2.4, 5.5


class Bird:
    def __init__(self, x, y, vx, wings, facing=1):
        self.x, self.y, self.vx, self.vy = x, y, vx, 0.0
        self.w = wings
        self.facing = facing
        self.cmd = (0.5, 0.0)
        self.phase = 0.0
        self.effort = 0.0

    def step(self, dt, wind=0.0):
        """Взмах даёт подъём и тягу; хвост вверх - больше подъёма, но и сопротивления; крыло гасит
        вертикальную скорость (VDAMP). Три команды с тягой под углом раскачивались до переполнения,
        и сеть не училась (замер) - осталось две."""
        flap, tail = self.cmd
        w = self.w
        air = max(0.1, self.vx - wind * self.facing)
        k = (air / w.cruise) ** 2
        ay = w.lift * k * (1.0 + 0.5 * tail) + w.up * flap - G - VDAMP * self.vy
        ax = w.push * flap - w.drag * k * (1.0 + 0.4 * tail * tail)
        self.vx = float(np.clip(self.vx + ax * dt, 0.3, 2.5 * w.cruise))
        self.vy = float(np.clip(self.vy + ay * dt, -15.0, 15.0))
        self.x += self.facing * self.vx * dt
        self.y += self.vy * dt
        self.phase = (self.phase + dt * (2.0 + 10.0 * flap)) % 1.0
        self.effort += flap * dt


class Flyer:
    """Сеть полёта породы (одна на породу мира, учится у каждой птицы)."""

    def __init__(self, seed=0):
        self.skill = GaussianSkill(4, [0.0, -1.0], [1.0, 1.0], hidden=16, seed=seed, sigma=0.25, lr=4e-3)
        last = self.skill.net.items[-1]
        last.b[:] = np.random.default_rng(seed + 9).normal(0.0, NEWBIE, last.b.shape).astype(np.float32)
        self.flights = 0

    def features(self, bird, height, target_h, wind):
        return np.array([np.clip((height - target_h) / 3.0, -2, 2), (bird.vx - bird.w.cruise) / bird.w.cruise,
                         np.clip(bird.vy / 3.0, -2, 2), wind / 5.0], np.float32)

    @staticmethod
    def reward(dh, dv, effort):
        # не ниже -3: улетевшая на десятки метров вверх давала -15 и раскачивала REINFORCE (замер)
        return max(-3.0, -abs(dh) / 2.0 - abs(dv) - 0.15 * effort)

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        self.flights = self.skill.attempts
        return self


def practice(flyer, look, rng, target_h=3.5, learn=True, seconds=FLIGHT):
    """Пробный полёт без жизни вокруг: старт на нужной высоте чуть медленнее крейсерской, ровный
    ветер. Возвращает средний промах по высоте, м (меньше - лучше)."""
    w = Wings(look)
    b = Bird(0.0, target_h, 0.7 * w.cruise, w)
    wind = float(rng.uniform(-2.0, 2.0))
    t, miss, n = 0.0, 0.0, 0
    while t < seconds:
        feats = flyer.features(b, b.y, target_h, wind)
        cmd, rec = flyer.skill.act(feats)
        b.cmd = tuple(float(c) for c in cmd)
        e0 = b.effort
        crashed = False
        for _ in range(int(DECIDE * 60)):
            b.step(1 / 60, wind)
            if b.y <= 0.0 or b.y > target_h + 10.0:
                crashed = True                   # села неуклюже или улетела ввысь - не то
                break
        t += DECIDE
        dh = b.y - target_h
        miss += abs(dh)
        n += 1
        if learn:
            r = CRASH if crashed else Flyer.reward(dh, (b.vx - w.cruise) / w.cruise, (b.effort - e0) / DECIDE)
            flyer.skill.learn(rec, r)
        if crashed:
            b.y, b.vy, b.vx = target_h, 0.0, 0.7 * w.cruise     # села неуклюже - снова взлетела
    flyer.flights += 1
    return miss / n
