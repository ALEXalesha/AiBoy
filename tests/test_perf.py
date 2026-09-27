"""Скорость: x8 в мире 1280 м укладывается в бюджет кадра (12 мс на жизнь), x16 - замер.

Процессор машины сборки часто занят другими задачами, поэтому берётся лучший из повторов:
закон - о том, сколько стоит шаг жизни, а не о соседях. Замер (план 2.0): 8 шагов в мире
1280 м - 3.0 мс в лучшем из 15 повторов при загрузке процессора 100 %, в мире 160 м - 2.8 мс.
Собака, птицы и игрушки добавятся к этому закону в своих задачах.
"""
import time

from game import models
from game.app import FRAME_BUDGET
from game.life import Life


def best_of(life, steps, repeats=15):
    best = 1e9
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(steps):
            life.tick()
        best = min(best, time.perf_counter() - t)
    return best


def test_x8_in_a_1280_m_world_fits_the_frame_budget():
    m = models.load()
    life = Life(m, 5, 7, 1280.0, models.starter_brain())
    for _ in range(120):
        life.tick()
    assert best_of(life, 8) < FRAME_BUDGET
