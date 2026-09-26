"""Музыкант: своя рекуррентная сеть (GRU из nn/, как у речи), играет по восьмым.

На каждом шаге сетки выход - четыре головы: удар или тишина; место на грифе (струна и лад
вместе - 6 x 13 = 78 мест); сила (3); глушить ли звучащие струны.

Струна и лад - одна голова, а не две, как в спеке. Замер: с двумя независимыми головами
сеть должна угадать согласованную пару (струна s, лад f: открытая s + f = нужная высота)
двумя отдельными выборками, и за 1500 попыток похожесть на «Чижика» росла только с 0.16 до
0.60, а на шаге Adam 3e-3 и выше обучение дважды срывалось в тишину.

Вход шага в GRU (IN чисел):

- последние 8 шагов его же игры: был ли удар, струна, лад/12, сила/2;
- где он в такте (8 мест) и номер такта во фразе;
- следующие 8 нот мелодии, которую он учит: есть ли, через сколько шагов, и на каком ладу
  она была бы на каждой из шести струн - (высота - открытая струна)/12; в свободной игре -
  нули;
- доля пройденной фразы и учит ли он мелодию.

Кроме GRU, у голов есть прямой вход - «нота мелодии начинается сейчас»: где на грифе она
звучит (78 мест, отмечены все, где можно) и сам факт начала. Его веса тоже учатся, с нуля;
это описание цели, а не правило: какое из мест взять и бить ли вообще, решает сеть.

Учится после каждой попытки одним шагом Adam (у прямого пути шаг крупнее: 0.03 против 0.001 -
замер: похожесть на «Чижика» к 500-й попытке 0.63-0.73 вместо 0.30): REINFORCE по всей фразе (сумма логарифмов
вероятностей выбранного, умноженная на «насколько попытка лучше обычной» - награда минус
скользящее среднее этой мелодии, делённое на её обычный разброс) и немного энтропии, чтобы
не застревать на одном.
"""
from dataclasses import dataclass

import numpy as np

from music import guitar
from nn import io
from nn.layers import Dense
from nn.losses import softmax
from nn.optim import Adam, clip_grads
from nn.rnn import GRU

HISTORY_STEPS = 8
AHEAD = 8
POSITIONS = guitar.STRINGS * guitar.FRETS
HEADS = (("strike", 2), ("pos", POSITIONS), ("force", guitar.FORCES), ("damp", 2))
OUT = sum(n for _, n in HEADS)
SLICES = {}
_k = 0
for _name, _n in HEADS:
    SLICES[_name] = slice(_k, _k + _n)
    _k += _n
PER_STEP = 1 + guitar.STRINGS + 2
PER_TARGET = 2 + guitar.STRINGS
IN = HISTORY_STEPS * PER_STEP + 8 + 1 + AHEAD * PER_TARGET + 2
NOW = POSITIONS + 1
HIDDEN = 96
LR = 1e-3
SKIP_LR = 0.03             # прямой путь «нота сейчас» - простая линейная связь, ему шаг крупнее
ENTROPY = 0.01
BASELINE_K = 0.1


def pos_of(string, fret):
    return string * guitar.FRETS + fret


def string_fret(pos):
    return divmod(int(pos), guitar.FRETS)


@dataclass
class Take:
    """Одна сыгранная фраза: входы и выбранное по шагам (для обучения)."""
    inputs: np.ndarray          # (T, IN) - вход GRU
    now: np.ndarray             # (T, NOW) - прямой вход «нота начинается сейчас»
    actions: np.ndarray         # (T, 4): удар, место, сила, глушить
    steps: int
    meter: int


def step_input(actions, t, steps, meter, target):
    x = np.zeros(IN, np.float32)
    k = 0
    for back in range(1, HISTORY_STEPS + 1):
        u = t - back
        if u >= 0 and actions[u, 0] == 1:
            s, f = string_fret(actions[u, 1])
            x[k] = 1.0
            x[k + 1 + s] = 1.0
            x[k + 1 + guitar.STRINGS] = f / 12.0
            x[k + 2 + guitar.STRINGS] = actions[u, 2] / 2.0
        k += PER_STEP
    x[k + (t % meter) % 8] = 1.0
    k += 8
    x[k] = (t // meter) / 4.0
    k += 1
    if target:
        ahead = [n for n in target if n.onset >= t][:AHEAD]
        for i, n in enumerate(ahead):
            base = k + i * PER_TARGET
            x[base] = 1.0
            x[base + 1] = (n.onset - t) / 8.0
            for s, o in enumerate(guitar.OPEN):
                x[base + 2 + s] = np.clip((n.pitch - o) / 12.0, -1.0, 2.0)
    k += AHEAD * PER_TARGET
    x[k] = t / max(steps, 1)
    x[k + 1] = 1.0 if target else 0.0
    return x


def now_input(t, target):
    z = np.zeros(NOW, np.float32)
    if target:
        for n in target:
            if n.onset == t:
                for s, f in guitar.positions(n.pitch):
                    z[pos_of(s, f)] = 1.0
                z[-1] = 1.0
    return z


def notes_of(actions, steps):
    """Выбранное по шагам -> ноты (шаг, струна, лад, сила, длительность в шагах): нота звучит
    до глушения, до нового удара по той же струне или до конца фразы."""
    out = []
    for t in range(steps):
        if actions[t, 0] != 1:
            continue
        s, f = string_fret(actions[t, 1])
        force = int(actions[t, 2])
        end = steps
        for u in range(t + 1, steps):
            if actions[u, 3] == 1 or (actions[u, 0] == 1 and string_fret(actions[u, 1])[0] == s):
                end = u
                break
        out.append((t, s, f, force, end - t))
    return out


class Musician:
    def __init__(self, seed=0, hidden=HIDDEN, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 7)
        self.dtype = dtype
        self.gru = GRU(IN, hidden, rng, dtype)
        self.out = Dense(hidden, OUT, rng, dtype, gain=0.5)
        self.out.b[SLICES["strike"]] = (0.0, -0.5)      # сначала бьёт реже, чем молчит
        self.out.b[SLICES["damp"]] = (1.0, 0.0)
        self.skip = Dense(NOW, OUT, rng, dtype)
        self.skip.W[...] = 0.0                           # прямой путь - с нуля, выучится сам
        self.opt = Adam(self.gru.params() + self.out.params(), lr=LR)
        self.opt_skip = Adam(self.skip.params(), lr=SKIP_LR)
        self.baseline = {}
        self.spread = {}            # разброс награды по мелодии - преимущество в его долях
        self.attempts = 0

    def nets(self):
        return {"gru": self.gru, "out": self.out, "skip": self.skip}

    def params(self):
        return [p for layer in self.nets().values() for p in layer.params()]

    def play(self, steps, meter, target=None, rng=None, greedy=False):
        rng = rng or self.rng
        h = np.zeros((1, self.gru.H), self.dtype)
        actions = np.zeros((steps, len(HEADS)), np.int64)
        inputs = np.zeros((steps, IN), self.dtype)
        nows = np.zeros((steps, NOW), self.dtype)
        for t in range(steps):
            inputs[t] = step_input(actions, t, steps, meter, target)
            nows[t] = now_input(t, target)
            h = self.gru.step(inputs[t][None], h)
            logits = (h @ self.out.W + self.out.b + nows[t][None] @ self.skip.W + self.skip.b)[0]
            for j, (name, n) in enumerate(HEADS):
                p = softmax(logits[SLICES[name]].astype(np.float64))
                actions[t, j] = int(np.argmax(p)) if greedy else int(rng.choice(n, p=p))
        return Take(inputs, nows, actions, steps, meter)

    def _loss_grad(self, take, advantage, entropy=None):
        """Потеря REINFORCE и градиент по логитам: -advantage * Σ log π(a) - entropy * Σ H.
        Место и сила учатся только на шагах с ударом."""
        entropy = ENTROPY if entropy is None else entropy
        X = take.inputs[None].astype(self.dtype)
        h0 = np.zeros((1, self.gru.H), self.dtype)
        hs = self.gru.forward(X, h0)
        logits = (self.out.forward(hs)[0] + self.skip.forward(take.now.astype(self.dtype))).astype(np.float64)
        grad = np.zeros_like(logits)
        loss = 0.0
        strike = take.actions[:, 0] == 1
        for j, (name, n) in enumerate(HEADS):
            sl = SLICES[name]
            p = softmax(logits[:, sl])
            logp = np.log(np.maximum(p, 1e-12))
            mask = strike.astype(np.float64) if name in ("pos", "force") else np.ones(len(p))
            a = take.actions[:, j]
            chosen = logp[np.arange(len(a)), a]
            H = -(p * logp).sum(axis=1)
            loss += float((-advantage * chosen * mask).sum() - entropy * (H * mask).sum())
            onehot = np.zeros_like(p)
            onehot[np.arange(len(a)), a] = 1.0
            g = -advantage * (onehot - p)                      # d(-adv·log p_a)/dz
            g += entropy * p * (logp + H[:, None])              # d(-ent·H)/dz
            grad[:, sl] = g * mask[:, None]
        return loss, grad

    def backward(self, grad):
        g = grad.astype(self.dtype)
        self.skip.backward(g)
        dhs = self.out.backward(g[None])
        self.gru.backward(dhs)

    def learn(self, take, reward, key="free"):
        """Один шаг после попытки. Возвращает использованное преимущество."""
        base = self.baseline.get(key, reward)
        var = self.spread.get(key, 0.05 ** 2)
        advantage = (reward - base) / (np.sqrt(var) + 1e-3)
        self.baseline[key] = base + BASELINE_K * (reward - base)
        self.spread[key] = var + BASELINE_K * ((reward - base) ** 2 - var)
        _, grad = self._loss_grad(take, advantage)
        self.backward(grad)
        clip_grads(self.params(), 5.0)
        self.opt.step()
        self.opt_skip.step()
        self.attempts += 1
        return advantage

    def save(self, path):
        arrays = io.collect(self.nets())
        keys = sorted(self.baseline)
        arrays["meta.baseline_keys"] = np.array(keys, dtype=str) if keys else np.zeros(0, str)
        arrays["meta.baseline"] = np.array([self.baseline[k] for k in keys], np.float64)
        arrays["meta.spread"] = np.array([self.spread.get(k, 0.05 ** 2) for k in keys], np.float64)
        arrays["meta.attempts"] = np.array([self.attempts])
        io.save(path, arrays)

    def load(self, path):
        arrays = io.load(path)
        io.restore(self.nets(), arrays)
        keys = [str(k) for k in arrays["meta.baseline_keys"]]
        self.baseline = {k: float(v) for k, v in zip(keys, arrays["meta.baseline"])}
        self.spread = {k: float(v) for k, v in zip(keys, arrays["meta.spread"])}
        self.attempts = int(arrays["meta.attempts"][0])
        for _, p, _ in self.params():
            if not np.isfinite(p).all():
                raise ValueError("в весах музыканта не числа")
        return self
