"""Сеть, которая придумывает мир: CPPN, как трасса в AiCar.

Две части одного генома:

- **рельеф** - вход: координата x в виде синусов и косинусов (длины волн от 80 до 10 м, с
  целым числом периодов на ширину мира - поэтому мир кольцом, без шва) и зерно мира из
  8 чисел; выход на каждое x - высота, оттенок земли, трава, плотность сущностей, логиты
  вида, цвет и размер сущности, дальние и средние горы, облака;
- **общее** - вход: только зерно; выход - палитра неба и земли, ночь, солнце, облачность,
  размах рельефа, щедрость на сущности, слоги имени.

Слои смешивают функции активации (sin, гаусс, tanh, «волна») - от этого узоры CPPN.
Веса не учатся градиентом, их отбирает эволюция (train/world.py) на интересность мира.
"""
import numpy as np

WAVELENGTHS = (80.0, 40.0, 24.0, 16.0, 10.0)
Z_DIM = 8
N_FEAT = 2 * len(WAVELENGTHS)

# выходы рельефа по x
T_HEIGHT, T_TONE, T_GRASS, T_DENSITY = 0, 1, 2, 3
T_KIND = slice(4, 10)
T_HUE, T_SIZE, T_FAR, T_MID, T_CLOUD, T_CLOUD_H = 10, 11, 12, 13, 14, 15
N_TERRAIN = 16
TERRAIN_LAYERS = (N_FEAT + Z_DIM, 20, 20, N_TERRAIN)

# общие выходы по зерну
G_SKY_TOP, G_SKY_BOTTOM, G_GROUND, G_GRASS = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)
G_NIGHT, G_SUN, G_CLOUDS, G_AMP, G_RICH, G_HUE_SHIFT = 12, 13, 14, 15, 16, 17
G_NAME = slice(18, 21)
N_GLOBAL = 21
GLOBAL_LAYERS = (Z_DIM, 16, N_GLOBAL)


def _gauss(x):
    return np.exp(-x * x)


def _wavelet(x):
    return x * np.exp(-x * x)


ACTS = (np.sin, _gauss, np.tanh, _wavelet)


def genome_size(layers):
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))


TERRAIN_SIZE = genome_size(TERRAIN_LAYERS)
GLOBAL_SIZE = genome_size(GLOBAL_LAYERS)
GENOME_SIZE = TERRAIN_SIZE + GLOBAL_SIZE


def _unpack(genome, layers):
    need = genome_size(layers)
    if len(genome) != need:
        raise ValueError(f"геном длины {len(genome)}, для слоёв {layers} нужно {need}")
    ws, bs, k = [], [], 0
    for i in range(len(layers) - 1):
        n_in, n_out = layers[i], layers[i + 1]
        ws.append(genome[k:k + n_in * n_out].reshape(n_in, n_out))
        k += n_in * n_out
        bs.append(genome[k:k + n_out])
        k += n_out
    return ws, bs


def _mixed(h):
    out = np.empty_like(h)
    for j in range(h.shape[1]):
        out[:, j] = ACTS[j % len(ACTS)](h[:, j])
    return out


def forward(genome, layers, x):
    """Прямой проход: скрытые слои со смесью активаций, выход - линейный (сырые числа,
    в метры и цвета их переводит generate.py)."""
    ws, bs = _unpack(genome, layers)
    h = x
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            h = _mixed(h)
    return h


def periods(width):
    """Сколько раз каждая волна укладывается в мир - целое, иначе был бы шов."""
    return np.array([max(1, round(width / L)) for L in WAVELENGTHS], float)


def features(x, width):
    phase = 2.0 * np.pi * np.outer(np.asarray(x, float) / width, periods(width))
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


def split(genome):
    return genome[:TERRAIN_SIZE], genome[TERRAIN_SIZE:]


def terrain(genome, x, width, z):
    t, _ = split(genome)
    f = features(x, width)
    inp = np.concatenate([f, np.broadcast_to(z, (len(f), Z_DIM))], axis=1)
    return forward(t, TERRAIN_LAYERS, inp)


def global_outputs(genome, z):
    _, g = split(genome)
    return forward(g, GLOBAL_LAYERS, np.asarray(z, float)[None, :])[0]


def random_genome(rng, scale=1.0):
    return rng.normal(0.0, scale, GENOME_SIZE)


def seed_vector(seed):
    """Зерно мира - 8 чисел из номера. Один номер - один мир."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, 0xA1B0]).normal(0.0, 1.0, Z_DIM)
