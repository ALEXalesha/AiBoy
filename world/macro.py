"""Макро-рельеф больших миров (640 и 1280 м): своя маленькая CPPN.

Сеть рельефа (world/cppn.py) видит волны 80 м и короче, поэтому в мире 1280 м один и тот
же узор холмов повторялся бы каждые 80 м. Эта сеть по зерну мира и координате (от одной до
семи волн на ширину, целое число - кольцо без шва) выдаёт три медленных числа:
- **изгиб** - сдвиг координаты, которую видит сеть рельефа (до ±WARP м): узор холмов
  растягивается и сжимается по-разному в разных местах, повтор пропадает;
- **высоту** - хребты, долины, большие котловины под озёра (до ±HEIGHT м);
- **размах** - где холмы выше, где ровнее (множитель 0.5-1.5).

Только для 640 и 1280: прежние размеры (80-320) остаются теми же байтами. Веса - эволюция
(train/macro.py) на «без видимого повтора и проходимо»; файл models/macro.npz.
"""
import numpy as np

from world.cppn import ACTS, Z_DIM, genome_size

# число волн на ширину мира: взаимно простые и разные для 640 и 1280 - иначе медленный
# рельеф сам повторялся бы (две волны по 320 м в мире 640 м - это два одинаковых хребта;
# замер: повтор до 0.64)
PERIODS = {640: (1, 2, 3, 5), 1280: (1, 3, 5, 7)}
N_FEAT = 2 * 4
LAYERS = (N_FEAT + Z_DIM, 12, 12, 3)
SIZE = genome_size(LAYERS)
WARP = 50.0             # замер: при 30 м мелкий узор в мирах 640 м местами ещё повторялся (0.56)
HEIGHT = 14.0
MIN_WIDTH = 640.0


def periods(width):
    key = 1280 if width >= 1280 else 640
    return np.array(PERIODS[key], float)


def _mixed(h):
    out = np.empty_like(h)
    for j in range(h.shape[1]):
        out[:, j] = ACTS[j % len(ACTS)](h[:, j])
    return out


def forward(genome, x, width, z):
    phase = 2.0 * np.pi * np.outer(np.asarray(x, float) / width, periods(width))
    inp = np.concatenate([np.sin(phase), np.cos(phase), np.broadcast_to(z, (len(phase), Z_DIM))], axis=1)
    k, h = 0, inp
    for i in range(len(LAYERS) - 1):
        n_in, n_out = LAYERS[i], LAYERS[i + 1]
        w = genome[k:k + n_in * n_out].reshape(n_in, n_out)
        k += n_in * n_out
        b = genome[k:k + n_out]
        k += n_out
        h = h @ w + b
        if i < len(LAYERS) - 2:
            h = _mixed(h)
    return h


def outputs(genome, x, width, z):
    """(изгиб м, высота м, размах) в точках x."""
    o = np.tanh(forward(genome, x, width, z))
    return WARP * o[:, 0], HEIGHT * o[:, 1], 1.0 + 0.5 * o[:, 2]


_default = None


def default_genome():
    """Веса из models/macro.npz (эволюция при сборке); нет файла - случайные по зерну 0
    (мир всё равно будет, только менее удачный)."""
    global _default
    if _default is None:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "macro.npz")
        try:
            with np.load(path) as data:
                g = np.asarray(data["genome"], float)
            if g.shape != (SIZE,) or not np.isfinite(g).all():
                raise ValueError("чужой файл")
        except Exception:  # noqa: BLE001
            g = np.random.default_rng(0).normal(0.0, 1.0, SIZE)
        _default = g
    return _default


def applies(width):
    return float(width) >= MIN_WIDTH


def repeat_score(heights, dx, lags=(80.0, 640.0)):
    """Насколько рельеф похож сам на себя со сдвигом: максимум нормированной автокорреляции
    по сдвигам от 80 до 640 м (1 - повтор один в один). Высоты - в отсчётах через dx."""
    h = np.asarray(heights, float)
    h = h - h.mean()
    f = np.fft.rfft(h)
    ac = np.fft.irfft(f * np.conj(f), len(h))
    ac /= ac[0] + 1e-12
    lo, hi = int(lags[0] / dx), min(int(lags[1] / dx), len(h) // 2)
    return float(ac[lo:hi + 1].max())


def local_repeat_score(heights, dx, lags=(80.0, 640.0), band=80.0):
    """То же для мелкого рельефа (без макро-высоты): высоты минус скользящее среднее по band м -
    повтор холмов виден глазом, даже если их несёт медленная волна."""
    h = np.asarray(heights, float)
    n = int(band / dx)
    k = np.ones(n) / n
    pad = np.concatenate([h[-n:], h, h[:n]])
    trend = np.convolve(pad, k, mode="same")[n:-n]
    return repeat_score(h - trend, dx, lags)


def shape_repeat_score(heights, dx, lags=(80.0, 640.0), window=40.0, band=80.0):
    """Повтор формы холмов, без высоты и размаха: куски по window м (без медленного рельефа,
    каждый приведён к среднему 0 и разбросу 1) сравниваются с кусками через сдвиг; средняя по
    кускам корреляция, максимум по сдвигам. Размах холмов, меняясь, не прячет одинаковую
    форму - этот замер её видит."""
    h = np.asarray(heights, float)
    n = int(band / dx)
    pad = np.concatenate([h[-n:], h, h[:n]])
    h = h - np.convolve(pad, np.ones(n) / n, mode="same")[n:-n]
    w = int(window / dx)
    starts = np.arange(0, len(h), w)
    idx = (starts[:, None] + np.arange(w)[None, :]) % len(h)
    lo, hi = int(lags[0] / dx), min(int(lags[1] / dx), len(h) // 2)
    step = max(1, int(1.0 / dx))

    def norm(a):
        a = a - a.mean(axis=1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    base = norm(h[idx])
    best = -1.0
    for lag in range(lo, hi + 1, step):
        other = norm(h[(idx + lag) % len(h)])
        best = max(best, float((base * other).sum(axis=1).mean()))
    return best
