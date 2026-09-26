"""Сводка того, что мозг видит и чего хочет, - вход речи и голоса.

Не скрытое состояние рекуррентного слоя: оно меняется, пока мозг учится на ходу, и речь,
обученная при сборке, перестала бы его понимать. Сводка - 27 понятных чисел от 0 до 1
(сторона и направление - от -1 до 1): холмы и ямы справа и слева, ровно ли вокруг,
ближайшая сущность (вид, сторона, расстояние, цвет), желания идти и прыгать, любопытство,
на земле ли, упал ли недавно, скорость, ночь, куда смотрит.
"""
import colorsys

import numpy as np

from brain.observe import nearest_entities
from world.generate import KINDS

FIELDS = (("hill_right", "hill_left", "pit_right", "pit_left", "flat")
          + tuple(f"kind_{k}" for k in KINDS)
          + ("entity", "entity_side", "entity_dist", "hue_cos", "hue_sin", "sat", "val",
             "want_dir", "want_jump", "curiosity", "grounded", "fell", "speed", "airborne",
             "night", "facing"))
STATE_DIM = len(FIELDS)
IDX = {name: i for i, name in enumerate(FIELDS)}
LOOK = 8.0            # м - холмы и ямы
SEE = 12.0            # м - сущности


def summarize(human, world, want_dir, want_jump, curiosity, fell):
    s = np.zeros(STATE_DIM, np.float32)
    x = human.px
    g0 = world.height_at(x)
    right = np.array([world.height_at(x + d) for d in np.arange(1.0, LOOK + 0.01, 0.5)]) - g0
    left = np.array([world.height_at(x - d) for d in np.arange(1.0, LOOK + 0.01, 0.5)]) - g0
    s[IDX["hill_right"]] = np.clip(right.max() / 2.5, 0, 1)
    s[IDX["hill_left"]] = np.clip(left.max() / 2.5, 0, 1)
    s[IDX["pit_right"]] = np.clip(-right.min() / 2.5, 0, 1)
    s[IDX["pit_left"]] = np.clip(-left.min() / 2.5, 0, 1)
    near = np.concatenate([left[:8], right[:8]])
    s[IDX["flat"]] = 1.0 - np.clip(near.std() / 0.6, 0, 1)
    ents = nearest_entities(human, world, n=1, see=SEE)
    if ents:
        dx, e = ents[0]
        s[IDX["kind_" + e.kind]] = 1.0
        s[IDX["entity"]] = 1.0
        s[IDX["entity_side"]] = 1.0 if dx >= 0 else -1.0
        s[IDX["entity_dist"]] = np.clip(abs(dx) / SEE, 0, 1)
        h, sat, val = colorsys.rgb_to_hsv(*e.color)
        s[IDX["hue_cos"]], s[IDX["hue_sin"]] = np.cos(2 * np.pi * h), np.sin(2 * np.pi * h)
        s[IDX["sat"]], s[IDX["val"]] = sat, val
    s[IDX["want_dir"]] = np.clip(want_dir, -1, 1)
    s[IDX["want_jump"]] = np.clip(want_jump, 0, 1)
    s[IDX["curiosity"]] = np.clip(curiosity, 0, 1)
    s[IDX["grounded"]] = 1.0 if human.grounded else 0.0
    s[IDX["fell"]] = np.clip(fell, 0, 1)
    s[IDX["speed"]] = np.clip(abs(human.vx) / 2.0, 0, 1)
    s[IDX["airborne"]] = 1.0 if (not human.grounded and human.py - g0 - human.body.leg() > 0.3) else 0.0
    s[IDX["night"]] = 1.0 if getattr(world, "night", False) else 0.0
    s[IDX["facing"]] = human.facing
    return s
