"""Мир по зерну: выходы CPPN переводятся в метры, цвета и сущности.

Код здесь ничего не выдумывает: высота, цвета, где стоят сущности, какого они вида,
цвета и размера - всё из выходов сети. Код только держит их в пределах (это защита,
чтобы мир нельзя было не разглядеть, а не правила, как ему выглядеть) и ставит сущности
в локальные максимумы плотности - без генератора случайных чисел.
"""
import colorsys
from dataclasses import dataclass

import numpy as np

from world import cppn, macro, mountains, reach

KINDS = ("tree", "bush", "stone", "flower", "mushroom", "crystal")
DX = 0.25                     # шаг высот, м
MIN_GAP = 2.0                 # сущности не ближе друг к другу, м
MAX_FLATTEN = 12              # раз по 15 % - рельеф в 7 раз ниже, дальше не надо
COMPANION = 5                 # отсчётов (1.25 м) - соседка у густого места
SYLLABLES = ("ла", "ми", "ро", "ка", "ну", "ти", "се", "во", "ра", "лу", "ни", "ко", "та",
             "зе", "ри", "мо", "са", "пе", "лё", "ю", "да", "фи", "ор", "эль")


def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(float(h) % 1.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1)))
    return (round(r, 4), round(g, 4), round(b, 4))


def mix(a, b, t):
    return tuple(round(float(x + (y - x) * t), 4) for x, y in zip(a, b))


@dataclass
class Entity:
    kind: str
    x: float
    y: float
    size: float
    color: tuple
    variant: float


@dataclass
class Cloud:
    x: float
    y: float
    size: float


def _smooth_ring(values, sigma):
    """Сглаживание по кольцу гауссовым окном (sigma - в отсчётах)."""
    n = len(values)
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    out = np.zeros(n)
    for off, wi in zip(k, w):
        out += wi * np.roll(values, off)
    return out


def _silhouette(values, sigma):
    """Силуэт гор: сглаженный выход сети, приведённый к размаху около 1 - чтобы горы
    были видны, даже если сеть выдала почти ровную линию (это защита, не форма)."""
    s = _smooth_ring(values, sigma)
    s = s - s.mean()
    return s / (s.std() + 0.15) * 0.9


def _local_maxima(values, threshold, positions, gap, width):
    """Индексы локальных максимумов выше порога; из близких остаётся более высокий."""
    left, right = np.roll(values, 1), np.roll(values, -1)
    cand = np.flatnonzero((values > left) & (values >= right) & (values > threshold))
    cand = cand[np.argsort(-values[cand])]
    chosen = []
    for i in cand:
        if all(min(abs(positions[i] - positions[j]), width - abs(positions[i] - positions[j])) >= gap
               for j in chosen):
            chosen.append(i)
    return sorted(chosen, key=lambda i: positions[i])


class World:
    def __init__(self, seed, width, heights, tone, grass, palette, night, sun_x, clouds_amount,
                 far, mid, clouds, entities, name):
        self.seed, self.width, self.dx = seed, float(width), DX
        self.heights, self.tone, self.grass = heights, tone, grass
        self.palette, self.night, self.sun_x, self.clouds_amount = palette, night, sun_x, clouds_amount
        self.far, self.mid, self.clouds, self.entities, self.name = far, mid, clouds, entities, name

    def heights_at(self, xs):
        """Высота земли в точках x (м), мир кольцом, между отсчётами - по прямой."""
        n = len(self.heights)
        u = (np.asarray(xs, float) % self.width) / self.dx
        i = np.floor(u).astype(int) % n
        t = u - np.floor(u)
        return self.heights[i] * (1 - t) + self.heights[(i + 1) % n] * t

    def height_at(self, x):
        """То же для одной точки - на чистом Python: физика зовёт это тысячи раз в секунду."""
        hl = self.__dict__.get("_hl")
        if hl is None:
            hl = self._hl = self.heights.tolist()
        u = (x % self.width) / self.dx
        i = int(u)
        t = u - i
        n = len(hl)
        return hl[i % n] * (1.0 - t) + hl[(i + 1) % n] * t

    def layer_at(self, layer, u):
        """Высота дальнего или среднего слоя гор в точке u (отсчёты через 1 м, кольцом)."""
        n = len(layer)
        u = np.asarray(u, float) % n
        i = np.floor(u).astype(int) % n
        t = u - np.floor(u)
        return layer[i] * (1 - t) + layer[(i + 1) % n] * t


def palette_from(g, night):
    o = sig(g)
    top_v = 0.08 + 0.14 * o[2] if night else 0.55 + 0.35 * o[2]
    bottom_v = 0.18 + 0.2 * o[5] if night else 0.78 + 0.2 * o[5]
    sky_top = hsv(o[0], 0.25 + 0.5 * o[1], top_v)
    sky_bottom = hsv(o[0] + 0.3 * (o[3] - 0.5), 0.15 + 0.45 * o[4], bottom_v)
    gv = (0.28 + 0.32 * o[8]) * (0.6 if night else 1.0)
    ground = hsv(o[6], 0.3 + 0.45 * o[7], gv)
    grass = hsv(o[6] + 0.3 * (o[9] - 0.5), 0.45 + 0.45 * o[10], min(1.0, gv + 0.12 + 0.2 * o[11]))
    ground_deep = hsv(o[6], 0.3 + 0.45 * o[7], gv * 0.5)
    return {"sky_top": sky_top, "sky_bottom": sky_bottom, "ground": ground, "ground_deep": ground_deep,
            "grass": grass, "far": mix(sky_bottom, ground, 0.3), "mid": mix(sky_bottom, ground, 0.6)}


def name_from(values):
    idx = [int(v * len(SYLLABLES)) % len(SYLLABLES) for v in sig(values)]
    parts = [SYLLABLES[i] for i in idx[:2]]
    if sig(values[2]) > 0.45:
        parts.append(SYLLABLES[idx[2]])
    word = "".join(parts)
    return word[0].upper() + word[1:]


def world_name(genome, seed):
    """Только имя мира - без рельефа: его считает маленькая общая часть сети."""
    return name_from(cppn.global_outputs(genome, cppn.seed_vector(seed))[cppn.G_NAME])


def generate(genome, seed, width, macro_genome=None):
    z = cppn.seed_vector(seed)
    g = cppn.global_outputs(genome, z)
    n = int(round(width / DX))
    xs = np.arange(n) * DX
    big = macro.applies(width)
    if big:
        # мир 640-1280 м: макро-сеть изгибает координату для сети рельефа (узор холмов не
        # повторяется через 80 м), задаёт хребты и котловины и где холмы выше
        warp, lift, scale = macro.outputs(macro_genome if macro_genome is not None else macro.default_genome(),
                                          xs, width, z)
        out = cppn.terrain(genome, xs + warp, width, z)
    else:
        out = cppn.terrain(genome, xs, width, z)

    amp = 1.2 + 3.3 * sig(g[cppn.G_AMP])
    # лёгкое сглаживание (полметра) - защита от зубцов, форму холмов задаёт сеть
    h = _smooth_ring(np.tanh(out[:, cppn.T_HEIGHT]), 2.0) * amp
    heights = np.clip(h - h.mean(), -6.0, 6.0)
    hills = None
    if big:
        hills = heights * scale            # холмы от сети рельефа - для замера повтора
        heights = hills + (lift - lift.mean())
    # защита от ям-ловушек: если откуда-то нельзя попасть во весь мир ни шагом, ни прыжком
    # (world/reach.py), рельеф той же формы делается ниже - на 15 % за раз. Сеть мира
    # отобрана так, что это редкость (2 мира из 60 на проверке), но закон - для всех миров.
    flattened = 0
    while flattened < MAX_FLATTEN and not reach.escapes(heights, DX).all():
        heights = heights * 0.85
        flattened += 1
    # горы (640 и 1280 м, world/mountains.py) - после защиты от ловушек: понижение их не трогает;
    # лёгкий путь проходим по построению, проверка - ниже и в законах
    relief = heights
    peaks = mountains.plan(width, seed) if big else []
    lowered = 0
    if peaks:
        # стена закрывает долине у подножия выход в одну сторону: если в другую мешает крутой
        # холм, это ловушка - холмы (не горы) ниже на 15 % за раз, как ямы выше
        smooth = mountains.smooth_ring(relief, DX, mountains.BASE)
        hills_now = relief
        while True:
            heights, _ = mountains.apply(hills_now, peaks, width, DX)
            if reach.escapes(heights, DX).all():
                break
            lowered += 1
            if lowered > MAX_FLATTEN:
                peaks, heights = [], relief  # не вышло проходимо - без гор (законы следят, чтобы не бывало)
                break
            hills_now = smooth + (hills_now - smooth) * 0.85
    tone = np.tanh(out[:, cppn.T_TONE])
    grass = sig(out[:, cppn.T_GRASS])

    night = bool(sig(g[cppn.G_NIGHT]) > 0.72)
    palette = palette_from(g, night)

    # сущности: максимумы плотности на сетке через 1 м (отсчёты 2, 6, 10... - это x = k + 0.5)
    idx = np.arange(2, n, 4)
    density = sig(out[idx, cppn.T_DENSITY])
    threshold = 0.62 - 0.3 * sig(g[cppn.G_RICH])
    hue_shift = sig(g[cppn.G_HUE_SHIFT])
    entities = []

    def entity_at(i, shrink=1.0):
        hue = float(sig(out[i, cppn.T_HUE]) + hue_shift)
        color = hsv(hue, 0.5 + 0.35 * sig(out[i, cppn.T_TONE]), 0.72 + 0.23 * sig(out[i, cppn.T_GRASS]))
        size = round(float(0.5 + 1.0 * sig(out[i, cppn.T_SIZE]) * shrink), 4)
        return Entity(KINDS[int(np.argmax(out[i, cppn.T_KIND]))], float(xs[i]), float(heights[i]), size, color,
                      round(float(sig(out[i, cppn.T_CLOUD_H])), 4))

    for k in _local_maxima(density, threshold, xs[idx], MIN_GAP, width):
        i = idx[k]
        entities.append(entity_at(i))
        # густое место - рядом ещё одна, поменьше: сторону и вид берёт из выходов сети там
        if density[k] > threshold + 0.12:
            side = 1 if out[i, cppn.T_TONE] > 0 else -1
            entities.append(entity_at((i + side * COMPANION) % n, shrink=0.6))
    entities.sort(key=lambda e: e.x)

    # дальние и средние горы: отсчёты через 1 м, сглаженные - это силуэты, а не рельеф
    ones = np.arange(0, n, 4)
    far = 4.0 + 5.0 * np.tanh(_silhouette(out[ones, cppn.T_FAR], 6.0))
    mid = 2.0 + 3.5 * np.tanh(_silhouette(out[ones, cppn.T_MID], 3.0))

    clouds_amount = float(sig(g[cppn.G_CLOUDS]))
    cidx = np.arange(0, n, 16)
    cd = sig(out[cidx, cppn.T_CLOUD])
    clouds = [Cloud(float(xs[cidx[k]]), float(6.0 + 5.0 * sig(out[cidx[k], cppn.T_CLOUD_H])),
                    float(1.0 + 2.5 * cd[k]))
              for k in _local_maxima(cd, 1.0 - 0.75 * clouds_amount, xs[cidx], 10.0, width)]

    world = World(seed, width, heights, tone, grass, palette, night, float(0.1 + 0.8 * sig(g[cppn.G_SUN])),
                  clouds_amount, far, mid, clouds, entities, name_from(g[cppn.G_NAME]))
    world.flattened = flattened
    world.hills = None if hills is None else hills * 0.85 ** flattened
    world.relief = relief                    # рельеф без гор - для замера повтора
    world.mountains = peaks
    world.mountain_lowered = lowered
    world.shallow_water = False
    if peaks:
        # глубокая вода - стена, пока он не пловец; у стены горы долину с другой стороны уже не
        # обойти - если глубокий пруд закрывает выход, вода в этом мире - вброд (замер: зажат
        # между прудом в 0.94 м, уступом и стеной горы)
        from world import water as wmod
        deep = [q for q in wmod.ponds(world) if q.level - q.bottom > wmod.WADE]
        if deep:
            walls = np.array(heights, float)
            xs_all = np.arange(n) * DX
            for q in deep:
                inside = ((xs_all - q.left) % width) <= ((q.right - q.left) % width)
                walls[inside] += 50.0
            if not reach.escapes(walls, DX).all():
                world.shallow_water = True
    return world
