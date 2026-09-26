"""Речь: посимвольная GRU на русском.

Вход - сводка состояния мозга (brain/state.py, 27 чисел). Она задаёт начальное скрытое
состояние (плотный слой и tanh) и приходит на каждый шаг вместе с прошлой буквой. Выход -
распределение следующей буквы. Фраза пишется по одной букве до знака конца или до
MAX_LEN. Училась сеть при сборке на фразах учителя (train/speech.py); в игре учителя нет,
только эта сеть и случайность выборки.
"""
import numpy as np

from brain.state import STATE_DIM
from nn import io
from nn.layers import Dense
from nn.losses import softmax, softmax_xent
from nn.optim import Adam, clip_grads
from nn.rnn import GRU
from speech import vocab

HIDDEN = 160


def one_hot(ids, size=vocab.SIZE, dtype=np.float32):
    out = np.zeros(ids.shape + (size,), dtype)
    np.put_along_axis(out, ids[..., None], 1.0, axis=-1)
    return out


class SpeechNet:
    def __init__(self, hidden=HIDDEN, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.hidden, self.dtype = hidden, dtype
        self.cond = Dense(STATE_DIM, hidden, rng, dtype, gain=1.0)
        self.gru = GRU(vocab.SIZE + STATE_DIM, hidden, rng, dtype)
        self.out = Dense(hidden, vocab.SIZE, rng, dtype, gain=1.0)
        self.opt = None

    def nets(self):
        return {"cond": self.cond, "gru": self.gru, "out": self.out}

    def params(self):
        return [p for layer in self.nets().values() for p in layer.params()]

    @staticmethod
    def batch(texts):
        """Входы (прошлые буквы, с началом) и цели (буквы, потом конец), маска длины."""
        seqs = [vocab.encode(t)[:vocab.MAX_LEN - 1] + [vocab.END] for t in texts]
        T = max(len(s) for s in seqs)
        target = np.zeros((len(seqs), T), np.int64)
        mask = np.zeros((len(seqs), T))
        for i, s in enumerate(seqs):
            target[i, :len(s)] = s
            mask[i, :len(s)] = 1.0
        prev = np.concatenate([np.full((len(seqs), 1), vocab.START), target[:, :-1]], axis=1)
        return prev, target, mask

    def loss_and_grads(self, states, texts):
        prev, target, mask = self.batch(texts)
        states = np.asarray(states, self.dtype)
        B, T = prev.shape
        x = np.concatenate([one_hot(prev, dtype=self.dtype),
                            np.broadcast_to(states[:, None, :], (B, T, STATE_DIM))], axis=2)
        h0 = np.tanh(self.cond.forward(states))
        hs = self.gru.forward(x, h0)
        logits = self.out.forward(hs)
        loss, g = softmax_xent(logits, target, mask)
        dhs = self.out.backward(g.astype(self.dtype))
        _, dh0 = self.gru.backward(dhs)
        self.cond.backward(dh0 * (1.0 - h0 * h0))
        return loss

    def train_batch(self, states, texts, lr=3e-3, clip=5.0):
        if self.opt is None:
            self.opt = Adam(self.params(), lr=lr)
        self.opt.lr = lr
        loss = self.loss_and_grads(states, texts)
        clip_grads(self.params(), clip)
        self.opt.step()
        return loss

    # --- в игре ---
    def _start(self, state):
        state = np.asarray(state, self.dtype)[None]
        return state, np.tanh(state @ self.cond.W + self.cond.b)

    def _next(self, state, h, prev_id):
        x = np.concatenate([one_hot(np.array([prev_id]), dtype=self.dtype), state], axis=1)
        h = self.gru.step(x, h)
        return h, (h @ self.out.W + self.out.b)[0]

    def sample(self, state, rng, temperature=0.7):
        state, h = self._start(state)
        ids, prev = [], vocab.START
        for _ in range(vocab.MAX_LEN):
            h, logits = self._next(state, h, prev)
            p = softmax(logits.astype(np.float64) / max(temperature, 1e-3))
            prev = int(rng.choice(len(p), p=p))
            if prev == vocab.END:
                break
            ids.append(prev)
        return vocab.decode(ids)

    def greedy(self, state):
        state, h = self._start(state)
        ids, prev = [], vocab.START
        for _ in range(vocab.MAX_LEN):
            h, logits = self._next(state, h, prev)
            prev = int(np.argmax(logits))
            if prev == vocab.END:
                break
            ids.append(prev)
        return vocab.decode(ids)

    def save(self, path):
        io.save(path, io.collect(self.nets()))

    def load(self, path):
        io.restore(self.nets(), io.load(path))
        return self
