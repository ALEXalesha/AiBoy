"""Рекуррентные слои на numpy.

RNNCell - один шаг простой рекуррентной сети, для мозга: он учится на ходу, и прошлое
состояние для него - просто вход (градиент в прошлое не идёт, усечение на один шаг).

GRU - по целой последовательности с обратным проходом во времени, для речи: сеть учится
при сборке на фразах целиком, а в игре пишет фразу по одной букве через step().
"""
import numpy as np

from nn.layers import he, sigmoid


class RNNCell:
    """h' = tanh(x Wx + h Wh + b)."""

    def __init__(self, nin, hidden, rng, dtype=np.float32):
        self.Wx = he(rng, nin, (nin, hidden), dtype, gain=1.0)
        self.Wh = he(rng, hidden, (hidden, hidden), dtype, gain=0.5)
        self.b = np.zeros(hidden, dtype)
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.db = np.zeros_like(self.b)

    def forward(self, x, h):
        self.x, self.h = x, h
        self.y = np.tanh(x @ self.Wx + h @ self.Wh + self.b)
        return self.y

    def backward(self, dy):
        da = dy * (1.0 - self.y * self.y)
        self.dWx[...] = self.x.T @ da
        self.dWh[...] = self.h.T @ da
        self.db[...] = da.sum(axis=0)
        return da @ self.Wx.T, da @ self.Wh.T

    def params(self):
        return [("Wx", self.Wx, self.dWx), ("Wh", self.Wh, self.dWh), ("b", self.b, self.db)]


class GRU:
    """z = σ(x Wz + h Uz), r = σ(x Wr + h Ur), n = tanh(x Wn + (r*h) Un), h' = (1-z) n + z h.

    Веса трёх ворот хранятся рядом: W (nin, 3H), U (H, 3H), b (3H) - порядок z, r, n."""

    def __init__(self, nin, hidden, rng, dtype=np.float32):
        H = hidden
        self.H = H
        self.W = he(rng, nin, (nin, 3 * H), dtype, gain=1.0)
        self.U = np.concatenate([np.linalg.qr(rng.normal(size=(H, H)))[0] for _ in range(3)],
                                axis=1).astype(dtype)
        self.b = np.zeros(3 * H, dtype)
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)
        self.db = np.zeros_like(self.b)

    def step(self, x, h):
        """Один шаг без запоминания - для выборки в игре."""
        H = self.H
        gx = x @ self.W + self.b
        gh = h @ self.U[:, :2 * H]
        z = sigmoid(gx[:, :H] + gh[:, :H])
        r = sigmoid(gx[:, H:2 * H] + gh[:, H:])
        n = np.tanh(gx[:, 2 * H:] + (r * h) @ self.U[:, 2 * H:])
        return (1.0 - z) * n + z * h

    def forward(self, x, h0):
        B, T, _ = x.shape
        H = self.H
        self.x, self.h0 = x, h0
        gx = (x @ self.W + self.b)
        self.cache = []
        out = np.empty((B, T, H), dtype=np.result_type(x, self.W))
        h = h0
        for t in range(T):
            gh = h @ self.U[:, :2 * H]
            z = sigmoid(gx[:, t, :H] + gh[:, :H])
            r = sigmoid(gx[:, t, H:2 * H] + gh[:, H:])
            rh = r * h
            n = np.tanh(gx[:, t, 2 * H:] + rh @ self.U[:, 2 * H:])
            h_new = (1.0 - z) * n + z * h
            self.cache.append((h, z, r, rh, n))
            out[:, t] = h_new
            h = h_new
        return out

    def backward(self, dout):
        B, T, _ = dout.shape
        H = self.H
        Uz, Ur, Un = self.U[:, :H], self.U[:, H:2 * H], self.U[:, 2 * H:]
        dgx = np.empty((B, T, 3 * H), dtype=dout.dtype)
        dU = np.zeros_like(self.U)
        dh = np.zeros((B, H), dtype=dout.dtype)
        for t in range(T - 1, -1, -1):
            h, z, r, rh, n = self.cache[t]
            dh = dh + dout[:, t]
            dn = dh * (1.0 - z)
            dz = dh * (h - n)
            dh_prev = dh * z
            dan = dn * (1.0 - n * n)
            dU[:, 2 * H:] += rh.T @ dan
            drh = dan @ Un.T
            dr = drh * h
            dh_prev += drh * r
            daz = dz * z * (1.0 - z)
            dar = dr * r * (1.0 - r)
            dU[:, :H] += h.T @ daz
            dU[:, H:2 * H] += h.T @ dar
            dh_prev += daz @ Uz.T + dar @ Ur.T
            dgx[:, t, :H] = daz
            dgx[:, t, H:2 * H] = dar
            dgx[:, t, 2 * H:] = dan
            dh = dh_prev
        nin = self.W.shape[0]
        self.dW[...] = self.x.reshape(-1, nin).T @ dgx.reshape(-1, 3 * H)
        self.dU[...] = dU
        self.db[...] = dgx.reshape(-1, 3 * H).sum(axis=0)
        return dgx @ self.W.T, dh

    def params(self):
        return [("W", self.W, self.dW), ("U", self.U, self.dU), ("b", self.b, self.db)]
