"""Сеть, которая придумывает человечка: маленький генератор по зерну.

Вход - зерно из 8 чисел, выход - 24 числа через сигмоиду: длины костей, толщины, цвета
кожи, волос, одежды и обуви, причёска, размер глаз, слоги имени. Скелет всегда
человеческий (15 точек, skeleton.py), сеть выбирает только размеры и цвета, а пределы
ниже - защита от уродцев, не правила, каким ему быть. Левая и правая рука (и нога) берут
одни и те же числа - человечек симметричный по устройству.

Веса не учатся: train/body.py отбирает из случайных геномов тот, у которого человечки
разных зёрен сильнее всего различаются.
"""
import colorsys
from dataclasses import dataclass

import numpy as np

from world.cppn import ACTS, genome_size

Z_DIM = 8
N_OUT = 24
LAYERS = (Z_DIM, 16, N_OUT)
GENOME_SIZE = genome_size(LAYERS)

# метры
LIMITS = {
    "head_r": (0.10, 0.14), "neck": (0.05, 0.09), "torso": (0.44, 0.60),
    "upper_arm": (0.25, 0.33), "forearm": (0.22, 0.30), "thigh": (0.38, 0.50), "shin": (0.36, 0.48),
    "arm_w": (0.065, 0.10), "leg_w": (0.085, 0.13), "torso_w": (0.20, 0.31), "eye": (0.8, 1.3),
}
# кожа - от светлой до тёмной, сеть выбирает место на шкале
SKIN = [(0.99, 0.88, 0.78), (0.95, 0.76, 0.62), (0.80, 0.58, 0.42), (0.60, 0.40, 0.27), (0.38, 0.25, 0.17)]
SYLLABLES = ("то", "ми", "бо", "ку", "ля", "ни", "па", "ти", "ро", "се", "ду", "ва", "ле", "му",
             "ка", "зи", "фо", "ю")


@dataclass(frozen=True)
class Body:
    head_r: float
    neck: float
    torso: float
    upper_arm: float
    forearm: float
    thigh: float
    shin: float
    arm_w: float
    leg_w: float
    torso_w: float
    eye: float
    skin: tuple
    hair: tuple
    shirt: tuple
    pants: tuple
    shoes: tuple
    hair_style: int
    cheeks: bool
    name: str

    def height(self):
        return self.shin + self.thigh + self.torso + self.neck + 2 * self.head_r

    def leg(self):
        return self.thigh + self.shin


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def _rgb(h, s, v):
    return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(float(h) % 1.0, float(s), float(v)))


def forward(genome, z):
    ws, bs, k = [], [], 0
    if len(genome) != GENOME_SIZE:
        raise ValueError(f"геном тела длины {len(genome)}, нужно {GENOME_SIZE}")
    for i in range(len(LAYERS) - 1):
        n_in, n_out = LAYERS[i], LAYERS[i + 1]
        ws.append(genome[k:k + n_in * n_out].reshape(n_in, n_out))
        k += n_in * n_out
        bs.append(genome[k:k + n_out])
        k += n_out
    h = np.asarray(z, float)[None, :] @ ws[0] + bs[0]
    h = np.stack([ACTS[j % len(ACTS)](h[:, j]) for j in range(h.shape[1])], axis=1)
    return sig(h @ ws[1] + bs[1])[0]


def seed_vector(seed):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, 0xB0D1]).normal(0.0, 1.0, Z_DIM)


def random_genome(rng, scale=1.2):
    return rng.normal(0.0, scale, GENOME_SIZE)


def _skin(t):
    t = float(np.clip(t, 0, 1)) * (len(SKIN) - 1)
    i = min(int(t), len(SKIN) - 2)
    f = t - i
    return tuple(round(a + (b - a) * f, 4) for a, b in zip(SKIN[i], SKIN[i + 1]))


def _hair(o_h, o_s, o_v):
    # чаще всего естественные цвета, изредка яркие - как сеть решит
    if o_s < 0.75:
        hue = 0.02 + 0.1 * o_h            # от чёрного через каштановый до светлого
        return _rgb(hue, 0.35 + 0.4 * o_s, 0.12 + 0.75 * o_v * o_v)
    return _rgb(o_h, 0.55 + 0.3 * o_s, 0.55 + 0.35 * o_v)


def generate_body(genome, seed):
    o = forward(genome, seed_vector(seed))
    vals = {}
    for i, (name, (lo, hi)) in enumerate(LIMITS.items()):
        vals[name] = round(float(lo + (hi - lo) * o[i]), 4)
    n = len(LIMITS)                                  # 11 чисел на размеры, дальше цвета
    shirt = _rgb(o[n + 3], 0.4 + 0.45 * o[n + 4], 0.45 + 0.45 * o[n + 5])
    pants = _rgb(o[n + 6], 0.25 + 0.5 * o[n + 7], 0.25 + 0.45 * o[n + 8])
    shoes = _rgb(o[n + 6] + 0.5 * o[n + 9], 0.3 + 0.4 * o[n + 9], 0.18 + 0.3 * o[n + 9])
    syl = [SYLLABLES[int(v * len(SYLLABLES)) % len(SYLLABLES)] for v in o[n + 10:n + 12]]
    name = "".join(syl)
    return Body(**vals, skin=_skin(o[n]), hair=_hair(o[n + 1], o[n + 2], o[n + 12]),
                shirt=shirt, pants=pants, shoes=shoes,
                hair_style=int(o[n + 11] * 997) % 4, cheeks=bool(o[n + 9] > 0.5),
                name=name[0].upper() + name[1:])
