"""Что ему по вкусу: своя маленькая сеть «признаки еды -> сколько удовольствия будет», учится
на каждом съеденном (средний квадрат, Adam). По ней он выбирает, к какой еде идти: softmax по
ожидаемому удовольствию минус цена дороги. Попробовал невкусное - сеть это запомнила, и такое
он выбирает реже (закон в тестах).
"""
import numpy as np

from nn import io
from nn.layers import Dense, Sequential, Tanh
from nn.losses import mse
from nn.optim import Adam
from world.food import N_FEAT

WAY_COST = 0.015             # удовольствия за метр дороги
T = 0.15                     # температура выбора еды
SEE = 40.0                   # м - о еде дальше он не думает
MEMORY = 60


class FoodPrefs:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.net = Sequential([Dense(N_FEAT, 12, rng, np.float64), Tanh(), Dense(12, 1, rng, np.float64, gain=0.1)])
        self.opt = Adam(self.net.params(), lr=0.01)
        self.meals = 0
        self.memory = []             # последние MEMORY съеденных: (признаки, удовольствие)

    def expect(self, feats):
        return float(np.tanh(self.net.forward(np.asarray(feats, np.float64)[None])[0, 0]))

    def learn(self, feats, pleasure, steps=40):
        """Съел - запомнил; сеть учится на всём, что помнит (иначе новое вкусное «переучивало»
        и старое невкусное - замер: после «плохое -0.9, хорошее 0.8» плохое выбиралось чаще)."""
        self.memory = (self.memory + [(np.asarray(feats, np.float64), float(pleasure))])[-MEMORY:]
        X = np.stack([f for f, _ in self.memory])
        Y = np.array([[np.arctanh(np.clip(p, -0.95, 0.95))] for _, p in self.memory])
        for _ in range(steps):
            _, g = mse(self.net.forward(X), Y)
            self.net.backward(g)
            self.opt.step()
        self.meals += 1

    def probabilities(self, options):
        """options - [(ключ, признаки, расстояние м)] -> {ключ: вероятность пойти туда}."""
        if not options:
            return {}
        z = np.array([self.expect(f) - WAY_COST * d for _, f, d in options]) / T
        z = np.exp(z - z.max())
        z /= z.sum()
        return {k: float(p) for (k, _, _), p in zip(options, z)}

    def save(self, path):
        arrays = io.collect({"net": self.net})
        arrays["meta.meals"] = np.array([self.meals])
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore({"net": self.net}, arrays)
        self.meals = int(arrays["meta.meals"][0])
        return self
