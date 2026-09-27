"""Навык «куда прилетит» (predict): по тому, как предмет летит сейчас, - где он будет на
высоте руки и через сколько. Нужен, чтобы ловить (catch): рука едет к предсказанной точке.

Учится на полётах, которые он видел, как руки - на своём опыте: увидел начало полёта
(скорость, высота над землёй, ветер) -> потом увидел, где предмет пересёк высоту руки и
когда. Никакой формулы баллистики в коде: только примеры, сеть и средний квадрат.

Вход: vx, vy (м/с, /10), высота над уровнем руки (м, /3), ветер (м/с, /5).
Выход: dx до точки встречи (м, /10) и время (с, /3).
"""
from collections import deque

import numpy as np

from nn import io
from nn.losses import mse
from nn.optim import Adam
from skills.base import mlp

IN, OUT = 4, 2


def features(vx, vy, height, wind=0.0):
    return np.array([vx / 10.0, vy / 10.0, height / 3.0, wind / 5.0], np.float32)


class Predictor:
    def __init__(self, seed=0, lr=3e-3, buffer=2000):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 11)
        self.net = mlp(IN, OUT, 48, rng, gain=0.5)
        self.opt = Adam(self.net.params(), lr=lr)
        self.buffer = deque(maxlen=buffer)
        self.flights = 0

    def predict(self, vx, vy, height, wind=0.0):
        y = self.net.forward(features(vx, vy, height, wind)[None])[0]
        return float(y[0] * 10.0), float(max(0.0, y[1] * 3.0))

    def saw(self, vx, vy, height, wind, dx, t, steps=30):
        """Увидел полёт целиком: пример в буфер и несколько шагов обучения."""
        self.buffer.append((features(vx, vy, height, wind), np.array([dx / 10.0, t / 3.0], np.float32)))
        self.flights += 1
        data = list(self.buffer)
        X = np.stack([d[0] for d in data])
        Y = np.stack([d[1] for d in data])
        for _ in range(steps):
            idx = self.rng.choice(len(data), min(64, len(data)), replace=False)
            _, g = mse(self.net.forward(X[idx]), Y[idx])
            self.net.backward(g)
            self.opt.step()

    def save(self, path):
        arrays = io.collect({"net": self.net})
        arrays["meta.flights"] = np.array([self.flights])
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore({"net": self.net}, arrays)
        self.flights = int(arrays["meta.flights"][0])
        return self
