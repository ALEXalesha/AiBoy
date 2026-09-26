"""Голос: плотная сеть от сводки состояния мозга к 7 параметрам синтезатора (synth.py).

Училась при сборке повторять учителя (train/voice.py): упал - низко и вниз, любопытно -
вверх, рядом сущность - три ноты, прыжок - коротко и высоко. В игре звучит, когда мозг
решил «звук».
"""
import numpy as np

from brain.state import STATE_DIM
from nn import io
from nn.layers import Dense, Sequential, Sigmoid, Tanh
from nn.losses import mse
from nn.optim import Adam
from voice.synth import PARAMS


class VoiceNet:
    def __init__(self, seed=0, hidden=24):
        rng = np.random.default_rng(seed)
        self.net = Sequential([Dense(STATE_DIM, hidden, rng, gain=1.0), Tanh(),
                               Dense(hidden, len(PARAMS), rng, gain=1.0), Sigmoid()])
        self.opt = None

    def train_batch(self, states, targets, lr=3e-3):
        if self.opt is None:
            self.opt = Adam(self.net.params(), lr=lr)
        self.opt.lr = lr
        loss, g = mse(self.net.forward(np.asarray(states, np.float32)), np.asarray(targets, np.float32))
        self.net.backward(g)
        self.opt.step()
        return loss

    def params_for(self, state):
        return self.net.forward(np.asarray(state, np.float32)[None])[0].astype(np.float64)

    def save(self, path):
        io.save(path, io.collect({"net": self.net}))

    def load(self, path):
        io.restore({"net": self.net}, io.load(path))
        return self
