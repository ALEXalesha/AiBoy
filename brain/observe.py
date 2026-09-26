"""Что видит мозг: 48 чисел, все - глазами человечка (впереди - куда он смотрит).

- рельеф: высоты в 11 точках от 4 м позади до 4 м впереди, от земли под ним;
- высота таза над землёй, скорость вперёд и вверх, какие стопы на земле, куда смотрит;
- свои 10 углов суставов;
- две ближайшие сущности: где (впереди/позади, выше/ниже), вид;
- лежит ли он после падения;
- «пульс» - синус и косинус своих часов тела с периодом PULSE. Рекуррентный слой учится
  с усечением на шаг и сам ритм не выучит; пульс даёт ритм, на который может опереться
  походка, - какой она будет и будет ли вообще, решает обучение.
"""
import math

import numpy as np

from body import skeleton
from world.generate import KINDS

OFFSETS = (-4.0, -3.0, -2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0, 3.0, 4.0)
TERRAIN = slice(0, len(OFFSETS))
SEE = 10.0                                  # м - дальше сущностей не видно
ENTITY = 3 + len(KINDS)
PULSE = 0.9                                 # с - период пульса
OBS_DIM = len(OFFSETS) + 6 + len(skeleton.JOINTS) + 2 * ENTITY + 1 + 2
CLOCK = slice(OBS_DIM - 2, OBS_DIM)
ENTITIES = slice(len(OFFSETS) + 6 + len(skeleton.JOINTS), len(OFFSETS) + 6 + len(skeleton.JOINTS) + 2 * ENTITY)
# «что увижу»: рельеф вокруг и сущности - то, что меняется, только когда он куда-то попал
WORLD = np.zeros(OBS_DIM, bool)
WORLD[TERRAIN] = True
WORLD[ENTITIES] = True


def nearest_entities(human, world, n=2, see=SEE):
    """Ближайшие сущности по кольцу мира: [(dx, сущность)], dx - в мировых координатах."""
    out = []
    width = world.width
    for e in getattr(world, "entities", ()):
        dx = (e.x - human.px + width / 2) % width - width / 2
        if abs(dx) <= see:
            out.append((dx, e))
    out.sort(key=lambda t: abs(t[0]))
    return out[:n]


def observe(human, world):
    f = human.facing
    g0 = world.height_at(human.px)
    o = np.zeros(OBS_DIM, np.float32)
    for i, off in enumerate(OFFSETS):
        o[i] = np.clip((world.height_at(human.px + f * off) - g0) / 2.0, -2.0, 2.0)
    k = len(OFFSETS)
    o[k] = np.clip((human.py - g0 - human.body.leg()) / 0.5, -2.0, 2.0)
    o[k + 1] = np.clip(human.vx * f / 3.0, -2.0, 2.0)
    o[k + 2] = np.clip(human.vy / 5.0, -2.0, 2.0)
    o[k + 3], o[k + 4] = float(human.contact[0]), float(human.contact[1])
    o[k + 5] = f
    k += 6
    o[k:k + len(skeleton.JOINTS)] = skeleton.to_unit(human.angles)
    k += len(skeleton.JOINTS)
    for dx, e in nearest_entities(human, world):
        o[k] = dx * f / SEE
        o[k + 1] = np.clip((e.y - g0) / 3.0, -2.0, 2.0)
        o[k + 2] = 1.0
        o[k + 3 + KINDS.index(e.kind)] = 1.0
        k += ENTITY
    o[CLOCK.start - 1] = 1.0 if human.fallen > 0 else 0.0
    phase = 2.0 * math.pi * human.clock / PULSE
    o[CLOCK.start], o[CLOCK.start + 1] = math.sin(phase), math.cos(phase)
    return o
