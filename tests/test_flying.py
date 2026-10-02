"""Ветер, пузыри, самолётик, бумеранг: физика и учёба."""
import math

import numpy as np
import pytest

import toys.flying as F
from physics.aero import Boomerang, Bubble, Plane
from world.weather import Wind


def test_wind_comes_from_the_weather_network_and_gusts():
    a, b = Wind(1), Wind(2)
    assert (a.base, a.periods) != (b.base, b.periods)
    xs = [a.at(0.0, 1.0, t) for t in np.arange(0, 60, 0.5)]
    assert np.std(xs) > 0.05 and max(abs(v) for v in xs) < 8.0


def test_a_bubble_drifts_faster_than_it_rises():
    b = Bubble(0.0, 1.5, 0.04)
    for _ in range(120):
        b.step(2.0, 1 / 60)
    assert b.vx > 3 * b.vy > 0


def fly(trim, angle=0.1, v=7.0):
    p = Plane(0.0, 1.6, v * math.cos(angle), v * math.sin(angle), trim)
    while p.y > 0 and p.time < 20:
        p.step(0.0, 1 / 60)
    return p.x, p.time


def test_an_airplane_with_zero_trim_glides_and_a_stalled_one_falls():
    t, x = 0.0, 0.0
    y, vx, vy = 1.6, 7 * math.cos(0.1), 7 * math.sin(0.1)
    while y > 0:                                         # камень, брошенный так же
        vy -= 9.81 / 60
        x += vx / 60
        y += vy / 60
        t += 1 / 60
    glide, _ = fly(0.0)
    stall, _ = fly(0.8)
    assert glide > 1.5 * x and stall < x
    # срыв: за критическим углом подъёмная сила падает, а не растёт (без этого 0.8 тоже падал -
    # от сопротивления, замер мутацией)
    p = Plane(0.0, 1.0, 7.0, 0.0)
    cl_stall, _ = p.coeffs(p.STALL)
    cl_past, _ = p.coeffs(p.STALL + 0.25)
    assert cl_past < 0.5 * cl_stall


def test_a_good_boomerang_throw_comes_back_within_1_5_m():
    b = Boomerang(0.0, 1.6, 5 * math.cos(0.7), 5 * math.sin(0.7), 60)
    closest = 99.0
    far = 0.0
    while b.time < 8 and b.y > 0:
        b.step(0.0, 1 / 60)
        far = max(far, b.home())
        if b.time > 1.0:
            closest = min(closest, b.home())
    assert far > 5.0 and closest < 1.5


@pytest.mark.parametrize("name,heads,attempt,wind,margin", [
    ("bubbles", F.BubbleHeads, F.BubbleAttempt, 3, 0.3),
    ("plane", F.PlaneHeads, F.PlaneAttempt, None, 0.15),
    ("boomerang", F.BoomerangHeads, F.BoomerangAttempt, None, 0.2),
])
def test_each_flying_toy_is_learned(name, heads, attempt, wind, margin):
    """Закон обучения, зерно 0, 300 попыток: медиана последних 50 выше первых 50 на запас.
    Замер (план): пузыри 0 -> 1.0 (ветреный мир), самолётик в штиль 0.29 -> 0.58, бумеранг
    0.03 -> 0.35-0.4."""
    h = heads(seed=0)
    rng = np.random.default_rng(0)
    w = Wind(wind) if wind is not None else None
    s = [F.practice(h, attempt, rng, wind=w) for _ in range(300)]
    assert np.median(s[-50:]) > np.median(s[:50]) + margin


def test_flying_toys_make_sounds():
    from sound import sfx
    for ev in ("whoosh", "pop", "land", "catch"):
        assert sfx.make(ev, 0.5, 0.05) is not None
