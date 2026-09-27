"""Общие головы-политики на numpy для занятий и игрушек.

- `ChoicePolicy` - выбор одного из нескольких действий: маленький перцептрон, softmax по
  разрешённым действиям, REINFORCE с базой (скользящее среднее награды) и нормировкой
  преимущества на скользящий разброс, плюс энтропия, чтобы не застывать;
- `ChoiceActorCritic` - то же для череды решений (навигатор): у каждого решения награда -
  то, что случилось до следующего решения, плюс оценка следующего состояния критиком
  (своя маленькая сеть). Без критика награда решения за 8 с вперёд почти вся зависела от
  следующих решений, и сеть не отличала «вперёд» от «назад» (замер в плане).

Так учился музыкант 1.1.0; здесь то же для любой головы, которой нужно выбрать из
нескольких вариантов (навигатор: вперёд / назад / осмотреться).
"""
import numpy as np

from nn import io
from nn.layers import Dense, Sequential, Tanh
from nn.optim import Adam, clip_grads

BASELINE_K = 0.02
ADV_CLIP = 5.0


def softmax_masked(logits, mask):
    z = np.where(mask, logits, -np.inf)
    z = z - z[mask].max()
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum()


class ChoicePolicy:
    def __init__(self, n_in, n_out, hidden=24, seed=0, lr=3e-3, entropy=0.1):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_out = n_in, n_out
        self.net = Sequential([Dense(n_in, hidden, rng, np.float64), Tanh(),
                               Dense(hidden, n_out, rng, np.float64, gain=0.1)])
        self.opt = Adam(self.net.params(), lr=lr)
        self.entropy = entropy
        self.baseline = 0.0
        self.spread = 1.0
        self.updates = 0
        self.rng = np.random.default_rng(seed + 1)

    def probs(self, x, mask=None):
        mask = np.ones(self.n_out, bool) if mask is None else np.asarray(mask, bool)
        logits = self.net.forward(np.asarray(x, np.float64)[None])[0]
        return softmax_masked(logits, mask)

    def choose(self, x, mask=None, greedy=False, rng=None):
        p = self.probs(x, mask)
        if greedy:
            return int(np.argmax(p)), p
        return int((rng or self.rng).choice(self.n_out, p=p)), p

    def learn(self, x, mask, action, reward):
        """Один шаг REINFORCE: действие action в состоянии x принесло reward."""
        mask = np.ones(self.n_out, bool) if mask is None else np.asarray(mask, bool)
        sd = np.sqrt(self.spread) + 1e-6
        dev = float(np.clip(reward - self.baseline, -ADV_CLIP * sd, ADV_CLIP * sd))
        adv = dev / sd
        self.baseline += BASELINE_K * dev
        self.spread += BASELINE_K * (dev * dev - self.spread)
        logits = self.net.forward(np.asarray(x, np.float64)[None])[0]
        p = softmax_masked(logits, mask)
        onehot = np.zeros(self.n_out)
        onehot[action] = 1.0
        logp = np.where(mask, np.log(np.maximum(p, 1e-12)), 0.0)
        h = -float((p * logp).sum())
        grad = -adv * (onehot - p) + self.entropy * p * (logp + h)
        grad = np.where(mask, grad, 0.0)
        self.net.backward(grad[None])
        clip_grads(self.net.params(), 1.0)
        self.opt.step()
        self.updates += 1
        return adv

    def save(self, path):
        arrays = io.collect({"net": self.net})
        arrays["meta.numbers"] = np.array([self.baseline, self.spread, self.updates], np.float64)
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore({"net": self.net}, arrays)
        b, s, u = arrays["meta.numbers"]
        self.baseline, self.spread, self.updates = float(b), float(s), int(u)
        for _, p, _ in self.net.params():
            if not np.isfinite(p).all():
                raise ValueError("в весах не числа")


class ChoiceActorCritic(ChoicePolicy):
    def __init__(self, n_in, n_out, hidden=24, seed=0, lr=3e-3, entropy=0.1, gamma=0.9, critic_lr=3e-3):
        super().__init__(n_in, n_out, hidden, seed, lr, entropy)
        rng = np.random.default_rng(seed + 7)
        self.critic = Sequential([Dense(n_in, hidden, rng, np.float64), Tanh(),
                                  Dense(hidden, 1, rng, np.float64, gain=0.1)])
        self.copt = Adam(self.critic.params(), lr=critic_lr)
        self.gamma = gamma

    def value(self, x):
        return float(self.critic.forward(np.asarray(x, np.float64)[None])[0, 0])

    def learn_step(self, x, mask, action, reward, x_next=None, steps=1.0):
        """Шаг актёра-критика: решение action в x принесло reward до следующего решения в
        x_next (None - череда оборвалась); steps - сколько обычных шагов решения прошло (у
        долгого действия будущее дисконтируется сильнее - иначе оно выгоднее просто тем,
        что длиннее)."""
        target = reward + (self.gamma ** steps * self.value(x_next) if x_next is not None else 0.0)
        v = self.critic.forward(np.asarray(x, np.float64)[None])
        delta = target - float(v[0, 0])
        self.critic.backward(np.array([[-delta]]))
        clip_grads(self.critic.params(), 1.0)
        self.copt.step()
        # актёр - по преимуществу delta, нормированному на скользящий разброс
        self.baseline = 0.0
        return self.learn(x, mask, action, delta)

    def save(self, path):
        arrays = io.collect({"net": self.net, "critic": self.critic})
        arrays["meta.numbers"] = np.array([self.baseline, self.spread, self.updates], np.float64)
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore({"net": self.net, "critic": self.critic}, arrays)
        b, s, u = arrays["meta.numbers"]
        self.baseline, self.spread, self.updates = float(b), float(s), int(u)
        for _, p, _ in self.net.params() + self.critic.params():
            if not np.isfinite(p).all():
                raise ValueError("в весах не числа")
