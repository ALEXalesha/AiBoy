"""Память мест: посещения, интерес, кольцо без шва, сводка «впереди / сзади», метки."""
import numpy as np
import pytest

from brain import places
from brain.places import Places


def walk(p, x0, x1, t0=0.0, interest=1.0, step=0.1):
    t = t0
    direction = 1 if x1 >= x0 else -1
    x = x0
    while (x - x1) * direction < 0:
        p.visit(x, t, interest)
        x += direction * step
        t += 0.1
    return t


def test_a_cell_counts_each_arrival_once_not_each_tick():
    p = Places(20)
    walk(p, 0.0, 5.0)
    assert list(p.visits[:5]) == [1, 1, 1, 1, 1] and p.visits[5:].sum() == 0
    walk(p, 5.0, 0.2)
    assert list(p.visits[1:5]) == [2, 2, 2, 2]


def test_the_ring_has_no_seam():
    p = Places(10)
    assert p.index(10.2) == p.index(0.2) and p.index(-0.5) == p.index(9.5)
    assert p.distance(9.0, 1.0, 1) == pytest.approx(2.0) and p.distance(1.0, 9.0, 1) == pytest.approx(8.0)
    assert p.distance(1.0, 9.0, -1) == pytest.approx(2.0)


def test_unknown_places_are_new_and_known_ones_less_so():
    p = Places(100)
    walk(p, 0.0, 10.0)
    n_ahead, _ = p.band(10.0, 1, 0, 6)
    n_behind, _ = p.band(10.0, -1, 0, 6)
    assert n_ahead == pytest.approx(1.0) and n_behind == pytest.approx(0.5)


def test_interest_is_relative_to_the_usual():
    p = Places(40)
    walk(p, 0.0, 10.0, interest=1.0)
    walk(p, 10.0, 12.0, interest=4.0)
    walk(p, 12.0, 20.0, interest=1.0)
    i = p.interest()
    assert i[10] > 2.0 and i[3] < 1.0 and i[30] == 1.0          # где не был - «как обычно»


def test_features_see_the_interesting_side_and_the_pull_of_a_mark():
    p = Places(200)
    walk(p, 0.0, 30.0, interest=1.0)
    walk(p, 30.0, 34.0, interest=5.0)          # что-то неожиданное
    walk(p, 34.0, 60.0, interest=1.0)
    f = p.features(45.0, 1)
    k = len(places.BANDS) * 2
    ahead_i = f[1:k:2]
    behind_i = f[k + 1:2 * k:2]
    assert behind_i[1] > ahead_i[1]            # 6-20 м сзади интереснее
    p.marks["гитара"] = 10.0
    f = p.features(45.0, 1, pulls={"гитара": 1.0})
    assert f[2 * k + 1] > f[2 * k] == 0.0      # гитара сзади
    f = p.features(45.0, -1, pulls={"гитара": 1.0})
    assert f[2 * k] > 0.0                      # развернулся - она впереди
    assert len(f) == places.N_FEATURES and np.isfinite(f).all()
