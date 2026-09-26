"""Руки для режима «Руками»: своя голова мозга на каждую руку - маленькая сеть «куда
тянуться -> команда плечу и локтю».

Учится как младенец учится тянуться - на своём опыте: команда (с небольшим шумом) ушла в
руку, кисть к моменту удара оказалась где-то; пара «где оказалась -> какая была команда» -
пример, и сеть учится по нему (средний квадрат ошибки, Adam, буфер последних примеров).
Так сеть сама узнаёт свою руку - вместе с тем, что рука не успевает доехать за шаг, - а
моторная награда «попал ли в точку» (расстояние кисти до цели) пишется в журнал и видна на
графике. Замер (в плане): REINFORCE по одной моторной награде за 600 попыток не поднялся
выше 4 % попаданий и потом разошёлся.

Вход: точка относительно плеча в долях длины руки и доля плеча в руке (пропорции у
человечков разные). Выход - в пределах суставов. Основной мозг ходьбы её не видит.
"""
from collections import deque

import numpy as np

from body import skeleton
from music import hands
from nn import io
from nn.layers import Dense, Sequential, Tanh
from nn.losses import mse
from nn.optim import Adam

SIGMA = 0.04          # рад - шум команды: чтобы опыт был разнообразным
LR = 3e-3
BUFFER = 600
EPOCHS = 4
BATCH = 64
IN = 3
ARMS = ("left", "right")
_SH = skeleton.JOINT_LIMITS[skeleton.JOINTS.index("sh_l")]
_EL = skeleton.JOINT_LIMITS[skeleton.JOINTS.index("el_l")]
LO = np.array([_SH[0], _EL[0]], np.float32)
HI = np.array([_SH[1], _EL[1]], np.float32)


def arm_input(body, lean, point):
    """Точка относительно плеча (в долях руки) и доля плеча в руке. Прошлую команду во вход
    добавлять пробовал (чтобы сеть учитывала, откуда рука поедет) - обучение на своём же
    выходе замкнулось: команда застыла, попаданий 0 из 600 попыток."""
    sx, sy = hands.shoulder(body, lean)
    L = body.upper_arm + body.forearm
    return np.array([(point[0] - sx) / L, (point[1] - sy) / L, body.upper_arm / L], np.float32)


def to_unit(a):
    return (np.asarray(a, np.float32) - LO) / (HI - LO) * 2.0 - 1.0


def from_unit(u):
    return LO + (HI - LO) * (np.clip(u, -1.0, 1.0) + 1.0) / 2.0


class Reacher:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 3)
        self.nets = {a: Sequential([Dense(IN, 48, rng), Tanh(), Dense(48, 2, rng, gain=0.3)]) for a in ARMS}
        self.opts = {a: Adam(self.nets[a].params(), lr=LR) for a in ARMS}
        self.buffer = {a: deque(maxlen=BUFFER) for a in ARMS}
        self.trace = {a: [] for a in ARMS}
        self.prev = {a: None for a in ARMS}
        self.explore = True

    def _rest(self, arm):
        i = (0, 1) if arm == "left" else (2, 3)
        rest = np.array(hands.SITTING, np.float32)[[2, 4, 3, 5]]
        return rest[list(i)]

    def aim(self, body, lean, target, arm):
        prev = self._rest(arm) if self.prev[arm] is None else self.prev[arm]
        x = arm_input(body, lean, target)
        a = from_unit(self.nets[arm].forward(x[None])[0])
        if self.explore:
            a = a + self.rng.normal(0.0, SIGMA, 2)
        a = np.clip(a, LO, HI)
        self.trace[arm].append((a, prev))
        self.prev[arm] = a
        return float(a[0]), float(a[1])

    def learn(self, attempt, body):
        """Опыт попытки -> примеры «где оказалась кисть -> команда», шаги Adam по буферу."""
        reached = getattr(attempt, "reached", [])
        lean = getattr(attempt, "lean", hands.SITTING[0])
        for k, arm in enumerate(ARMS):
            for (cmd, prev), pts in zip(self.trace[arm], reached):
                self.buffer[arm].append((arm_input(body, lean, pts[k]), to_unit(cmd)))
            data = list(self.buffer[arm])
            if not data:
                continue
            X = np.stack([d[0] for d in data])
            Y = np.stack([d[1] for d in data])
            for _ in range(EPOCHS):
                idx = self.rng.choice(len(data), min(BATCH, len(data)), replace=False)
                _, g = mse(self.nets[arm].forward(X[idx]), Y[idx])
                self.nets[arm].backward(g)
                self.opts[arm].step()
        self.trace = {a: [] for a in ARMS}
        self.prev = {a: None for a in ARMS}

    def save(self, path):
        io.save(path, io.collect(self.nets))

    def load(self, path):
        io.restore(self.nets, io.load(path))
        return self
