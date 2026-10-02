"""Два способа учить навык.

**ExperienceSkill** - на своём опыте. Команда (с небольшим шумом) ушла в тело, получилось
что-то; пара «что получилось -> какая была команда» - пример, сеть учится по нему (средний
квадрат, Adam, буфер последних примеров). Так сеть сама узнаёт своё тело. Так учатся руки
1.1.0: REINFORCE по моторной награде там за 600 попыток не поднялся выше 4 % попаданий.

**GaussianSkill** - REINFORCE: команда = выход сети + гауссов шум; после попытки награда,
шаг по (награда - база) / разброс. Для навыков, где «что получилось» нельзя обратить в
команду (удар ногой: важна не точка, а дальность и точность вместе).

Выход обоих - в пределах lo..hi (сеть выдаёт -1..1).
"""
from collections import deque

import numpy as np

from nn import io
from nn.layers import Dense, Sequential, Tanh
from nn.losses import mse
from nn.optim import Adam, clip_grads


def mlp(n_in, n_out, hidden, rng, gain=0.3):
    return Sequential([Dense(n_in, hidden, rng), Tanh(), Dense(hidden, n_out, rng, gain=gain)])


class Bounds:
    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)

    def to_unit(self, a):
        return (np.asarray(a, np.float32) - self.lo) / (self.hi - self.lo) * 2.0 - 1.0

    def from_unit(self, u):
        return self.lo + (self.hi - self.lo) * (np.clip(u, -1.0, 1.0) + 1.0) / 2.0


class ExperienceSkill:
    def __init__(self, n_in, lo, hi, hidden=48, seed=0, sigma=0.04, lr=3e-3, buffer=600, epochs=4, batch=64):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 3)
        self.bounds = Bounds(lo, hi)
        self.net = mlp(n_in, len(self.bounds.lo), hidden, rng)
        self.opt = Adam(self.net.params(), lr=lr)
        self.buffer = deque(maxlen=buffer)
        self.sigma, self.epochs, self.batch = sigma, epochs, batch
        self.explore = True

    def act(self, x):
        a = self.bounds.from_unit(self.net.forward(np.asarray(x, np.float32)[None])[0])
        if self.explore:
            a = a + self.rng.normal(0.0, self.sigma, len(a)) * (self.bounds.hi - self.bounds.lo) / 2.0
        return np.clip(a, self.bounds.lo, self.bounds.hi)

    def remember(self, outcome_x, command):
        """Пример: команда command привела туда, что описывает outcome_x (в тех же единицах,
        что вход сети)."""
        self.buffer.append((np.asarray(outcome_x, np.float32), self.bounds.to_unit(command)))

    def learn(self):
        data = list(self.buffer)
        if not data:
            return None
        X = np.stack([d[0] for d in data])
        Y = np.stack([d[1] for d in data])
        loss = None
        for _ in range(self.epochs):
            idx = self.rng.choice(len(data), min(self.batch, len(data)), replace=False)
            loss, g = mse(self.net.forward(X[idx]), Y[idx])
            self.net.backward(g)
            self.opt.step()
        return float(loss)

    def nets(self):
        return {"net": self.net}

    def save(self, path):
        io.save(path, io.collect(self.nets()))

    def load(self, path):
        io.restore(self.nets(), io.load(path))
        return self


class GaussianSkill:
    def __init__(self, n_in, lo, hi, hidden=32, seed=0, sigma=0.15, lr=3e-3, baseline_k=0.05):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 5)
        self.bounds = Bounds(lo, hi)
        self.net = mlp(n_in, len(self.bounds.lo), hidden, rng, gain=0.1)
        self.opt = Adam(self.net.params(), lr=lr)
        self.sigma = sigma                   # шум в долях -1..1
        self.baseline, self.spread, self.k = 0.0, 1.0, baseline_k
        self.explore = True
        self.attempts = 0

    def act(self, x):
        """(команда, запись для learn)."""
        x = np.asarray(x, np.float32)
        mu = np.tanh(self.net.forward(x[None])[0])
        u = mu + (self.rng.normal(0.0, self.sigma, len(mu)) if self.explore else 0.0)
        u = np.clip(u, -1.0, 1.0)
        return self.bounds.from_unit(u), (x, u)

    def learn(self, record, reward):
        x, u = record
        sd = np.sqrt(self.spread) + 1e-6
        dev = float(np.clip(reward - self.baseline, -5 * sd, 5 * sd))
        adv = dev / sd
        self.baseline += self.k * dev
        self.spread += self.k * (dev * dev - self.spread)
        z = self.net.forward(x[None])
        mu = np.tanh(z)
        # -d(log N(u; mu, sigma))/dmu * adv, через tanh
        dmu = -adv * (u[None] - mu) / (self.sigma * self.sigma)
        self.net.backward((dmu * (1.0 - mu * mu)).astype(np.float32))
        clip_grads(self.net.params(), 1.0)
        self.opt.step()
        self.attempts += 1
        return adv

    def learn_many(self, records, reward):
        """Череда решений одной попытки (поездка, башня) и одна награда на всех: база и разброс
        сдвигаются один раз, а не на каждое решение (иначе к десятому решению преимущество уже
        ноль - замер: велосипед находил езду и терял её)."""
        if not records:
            return 0.0
        sd = np.sqrt(self.spread) + 1e-6
        dev = float(np.clip(reward - self.baseline, -5 * sd, 5 * sd))
        adv = dev / sd
        self.baseline += self.k * dev
        self.spread += self.k * (dev * dev - self.spread)
        X = np.stack([r[0] for r in records])
        U = np.stack([r[1] for r in records])
        mu = np.tanh(self.net.forward(X))
        dmu = -adv * (U - mu) / (self.sigma * self.sigma) / len(records)
        self.net.backward((dmu * (1.0 - mu * mu)).astype(np.float32))
        clip_grads(self.net.params(), 1.0)
        self.opt.step()
        self.attempts += 1
        return adv

    def nets(self):
        return {"net": self.net}

    def save(self, path):
        arrays = io.collect(self.nets())
        arrays["meta.numbers"] = np.array([self.baseline, self.spread, self.attempts], np.float64)
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore(self.nets(), arrays)
        b, s, n = arrays["meta.numbers"]
        self.baseline, self.spread, self.attempts = float(b), float(s), int(n)
        return self
