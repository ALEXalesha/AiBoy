"""Вода для предметов pymunk (мяч, кубики, поплавок, собака) - своя: в pymunk её нет.

Каждый шаг для тела в воде: выталкивающая сила по доле под водой (Архимед: вес вытесненной
воды), сопротивление (линейное и квадратичное, против скорости) и вращения; всплеск - волна
на поверхности, если тело вошло в воду быстро. Волны - маленькая струна на поверхности
каждого водоёма: смещение по точкам, затухание; рисуются линией поверхности.

Плотность тела - масса / площадь (мы в 2D: площадь фигуры на метр толщины); воды - RHO.
Тело плотнее воды тонет, легче - плавает на своей доле (закон в тестах).
"""
import math

import numpy as np

RHO = 1000.0                 # кг/м³ (на метр толщины - кг/м²)
G = 9.81
DRAG_LIN = 5.0               # 1/с - линейное сопротивление в воде
DRAG_QUAD = 1.2              # квадратичное
SPIN_DRAG = 2.5
WAVE_DX = 0.25               # м между точками волны
WAVE_DAMP = 0.6              # 1/с
WAVE_C = 1.2                 # м/с - скорость волны
SPLASH_V = 1.5               # м/с - быстрее вошёл - всплеск


def circle_submerged(r, depth):
    """Площадь круга радиуса r ниже уровня, если центр на depth выше уровня (depth < 0 - ниже)."""
    h = np.clip(r - depth, 0.0, 2 * r)           # высота погружённого сегмента
    if h <= 0:
        return 0.0
    if h >= 2 * r:
        return math.pi * r * r
    return r * r * math.acos((r - h) / r) - (r - h) * math.sqrt(max(0.0, 2 * r * h - h * h))


def box_submerged(w, hgt, depth):
    """Площадь прямоугольника w x hgt (без поворота) ниже уровня; центр на depth выше уровня."""
    sub = np.clip(hgt / 2 - depth, 0.0, hgt)
    return w * sub


class Waves:
    def __init__(self, left, right):
        self.left = left
        self.n = max(3, int((right - left) / WAVE_DX) + 1)
        self.y = np.zeros(self.n)
        self.v = np.zeros(self.n)

    def splash(self, x, strength):
        i = int(np.clip((x - self.left) / WAVE_DX, 1, self.n - 2))
        self.v[i] -= strength

    def step(self, dt):
        lap = np.zeros(self.n)
        lap[1:-1] = self.y[:-2] - 2 * self.y[1:-1] + self.y[2:]
        k = (WAVE_C / WAVE_DX) ** 2
        self.v += (k * lap - WAVE_DAMP * self.v) * dt
        self.y += self.v * dt
        self.y[0] = self.y[-1] = 0.0

    def at(self, x):
        i = int(np.clip((x - self.left) / WAVE_DX, 0, self.n - 1))
        return float(self.y[i])


def apply(body, shape_kind, size, level, dt, waves=None):
    """Сила воды на тело pymunk за шаг. shape_kind: "circle" (size - радиус) или "box" ((w, h)).
    Возвращает долю под водой."""
    x, y = body.position
    surface = level + (waves.at(x) if waves is not None else 0.0)
    depth = y - surface
    if shape_kind == "circle":
        area = math.pi * size * size
        sub = circle_submerged(size, depth)
    else:
        w, h = size
        area = w * h
        sub = box_submerged(w, h, depth)
    share = sub / area if area else 0.0
    if share <= 0:
        body._was_wet = False
        return 0.0
    vx, vy = body.velocity
    if waves is not None and not getattr(body, "_was_wet", False) and vy < -SPLASH_V:
        waves.splash(x, min(2.0, -vy * 0.2))
    body._was_wet = True
    buoy = RHO * sub * G
    body.apply_force_at_world_point((0.0, buoy), (x, y))
    speed = math.hypot(vx, vy)
    k = share * (DRAG_LIN + DRAG_QUAD * speed) * body.mass
    body.velocity = (vx * max(0.0, 1.0 - k / body.mass * dt), vy * max(0.0, 1.0 - k / body.mass * dt))
    body.angular_velocity *= max(0.0, 1.0 - SPIN_DRAG * share * dt)
    return share
