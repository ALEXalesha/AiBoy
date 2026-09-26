"""Слои сети на numpy (из TicTacToeAi и нейро-змейки, плюс сигмоида).

У каждого слоя forward(x) -> y, backward(dy) -> dx (градиенты весов пишутся в self.d*),
params() -> [(имя, массив, градиент)] - по этому списку работают Adam и сохранение.
"""
import numpy as np


def he(rng, fan_in, shape, dtype, gain=2.0):
    return (rng.normal(size=shape) * np.sqrt(gain / fan_in)).astype(dtype)


class Dense:
    """Плотный слой по последней оси: работает и на (B, n), и на (B, T, n)."""

    def __init__(self, nin, nout, rng, dtype=np.float32, gain=2.0):
        self.W = he(rng, nin, (nin, nout), dtype, gain)
        self.b = np.zeros(nout, dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        n = self.W.shape[0]
        self.dW[...] = self.x.reshape(-1, n).T @ dy.reshape(-1, dy.shape[-1])
        self.db[...] = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        return dy @ self.W.T

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask

    def params(self):
        return []


class Tanh:
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dy):
        return dy * (1.0 - self.y * self.y)

    def params(self):
        return []


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Sigmoid:
    """Выход в (0, 1): параметры звука, доли, вероятности."""

    def forward(self, x):
        self.y = sigmoid(x)
        return self.y

    def backward(self, dy):
        return dy * self.y * (1.0 - self.y)

    def params(self):
        return []


class Sequential:
    """Слои по порядку. Имена весов - номер слоя и имя в нём: '0.W', '2.b'."""

    def __init__(self, items):
        self.items = list(items)

    def forward(self, x):
        for layer in self.items:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.items):
            dy = layer.backward(dy)
        return dy

    def params(self):
        return [(f"{i}.{name}", p, g) for i, layer in enumerate(self.items)
                for name, p, g in layer.params()]
