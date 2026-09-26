"""Вкус владельца: маленькая логистическая сеть по признакам попытки.

Владелец ставит 👍 или 👎 - это самая весомая часть награды (C). Когда его нет рядом,
модель вкуса угадывает, что бы он поставил: учится на всех его оценках этого человечка
(признаки попытки -> понравилось ли), каждый раз заново, двумя сотнями шагов Adam - оценок
немного. Догадка - число от -1 до 1; без оценок мнения нет - 0.

Точность считается честно: догадка делается до того, как модель увидела эту оценку.
"""
import numpy as np

from music import reward
from nn.layers import Dense, sigmoid
from nn.losses import bce
from nn.optim import Adam

FEATURES = 12
STEPS = 200
L2 = 1e-3


def features(notes, closeness, steps=32, meter=8):
    """Признаки попытки: высота (средняя, размах, доля высоких), густота, сила, длительность,
    аккорды, консонанс, ритм, похожесть на мелодию (если учил) и учил ли."""
    f = np.zeros(FEATURES, np.float32)
    if notes:
        pitches = np.array([n.pitch for n in notes], float)
        f[0] = (pitches.mean() - 58.0) / 18.0
        f[1] = (pitches.max() - pitches.min()) / 24.0
        f[2] = float((pitches >= 64).mean())
        f[3] = len(notes) / max(steps, 1)
        f[4] = float(np.mean([n.force for n in notes])) / 2.0
        f[5] = float(np.mean([n.dur for n in notes])) / 8.0
        onsets = [n.onset for n in notes]
        f[6] = 1.0 - len(set(onsets)) / len(onsets)
        f[7] = reward.consonance(notes)
        f[8] = reward.rhythm(notes, meter)
    f[9] = 0.0 if closeness is None else closeness
    f[10] = 0.0 if closeness is None else 1.0
    f[11] = 1.0                                       # постоянный признак
    return np.clip(f, -3.0, 3.0)


TRUST_N = 10            # при стольких оценках доверие - половина
TRUST_MIN_GUESSES = 5   # после стольких догадок доверие зависит от точности
COIN = 0.6              # точность не выше - доверия нет
GOOD = 0.8              # точность не ниже - доверие полное


class Taste:
    def __init__(self):
        self.ratings = []           # [(признаки списком, +1 или -1)]
        self.guesses = 0
        self.hits = 0
        self.net = None

    @classmethod
    def from_ratings(cls, ratings, guesses=0, hits=0):
        t = cls()
        t.ratings = [(list(map(float, f)), int(r)) for f, r in ratings]
        t.guesses, t.hits = int(guesses), int(hits)
        t._fit()
        return t

    def _fit(self):
        if not self.ratings:
            self.net = None
            return
        rng = np.random.default_rng(0)
        net = Dense(FEATURES, 1, rng, np.float64, gain=0.1)
        opt = Adam(net.params(), lr=0.05)
        X = np.array([f for f, _ in self.ratings], np.float64)
        y = np.array([[1.0 if r > 0 else 0.0] for _, r in self.ratings])
        for _ in range(STEPS):
            _, g = bce(net.forward(X), y)
            net.backward(g)
            net.dW += L2 * net.W
            opt.step()
        self.net = net

    def predict(self, feats):
        """Догадка об оценке владельца: от -1 (не понравится) до 1 (понравится)."""
        if self.net is None:
            return 0.0
        p = float(sigmoid(np.asarray(feats, np.float64)[None] @ self.net.W + self.net.b)[0, 0])
        return 2.0 * p - 1.0

    def rate(self, feats, rating):
        """Оценка владельца: сначала догадка (для точности), потом учимся на ней."""
        if self.net is not None:
            self.guesses += 1
            self.hits += (self.predict(feats) > 0) == (rating > 0)
        self.ratings.append((list(map(float, feats)), 1 if rating > 0 else -1))
        self._fit()

    def accuracy(self):
        return None if not self.guesses else self.hits / self.guesses

    def trust(self):
        """Насколько верить догадке в награде, 0..1: пока вкус не угадал хотя бы
        TRUST_MIN_GUESSES оценок - 0; дальше растёт с числом оценок и падает до 0, если вкус
        угадывает не лучше монетки. Замер: без этого восемь случайных оценок за 300 попыток
        роняли похожесть на мелодию до 0 - догадка с самым большим весом перевешивала; с
        доверием по числу оценок, но без проверки точности (0.1-0.3) - тоже."""
        if self.guesses < TRUST_MIN_GUESSES:
            return 0.0
        n = len(self.ratings)
        return n / (n + TRUST_N) * float(np.clip((self.accuracy() - COIN) / (GOOD - COIN), 0.0, 1.0))
