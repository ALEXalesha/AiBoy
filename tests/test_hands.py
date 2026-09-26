import math

import numpy as np
import pytest

from body.cppn import generate_body, random_genome
from music import guitar, hands


@pytest.fixture(scope="module")
def bodies():
    g = random_genome(np.random.default_rng(2))
    return [generate_body(g, s) for s in range(6)]


def test_frets_get_closer_towards_the_body(bodies):
    b = bodies[0]
    gaps = [hands.fret_spacing(k, b) for k in range(1, guitar.FRETS)]
    assert all(a > c for a, c in zip(gaps, gaps[1:]))
    assert hands.fret_distance(12, b) == pytest.approx(hands.SCALE_M * b.height() / 1.7 / 2)


def test_every_fret_and_string_is_reachable_by_the_arm(bodies):
    for b in bodies:
        lean = hands.SITTING[0]
        for s in range(guitar.STRINGS):
            for f in range(guitar.FRETS):
                t = hands.left_target(s, f, b)
                sh, el = hands.reach(b, lean, t)
                assert math.dist(hands.hand(b, lean, sh, el), t) < 0.01, (b.name, s, f)
            t = hands.right_target(s, b)
            sh, el = hands.reach(b, lean, t)
            assert math.dist(hands.hand(b, lean, sh, el), t) < 0.01


def test_hit_means_the_right_fret_and_string(bodies):
    b = bodies[1]
    t = hands.left_target(2, 5, b)
    assert hands.left_hit(t, 2, 5, b)
    assert not hands.left_hit(hands.left_target(2, 7, b), 2, 5, b)
    assert not hands.left_hit(hands.left_target(4, 5, b), 2, 5, b)
    assert hands.left_hit((9.0, 9.0), 3, 0, b)                  # открытая струна
    assert hands.right_hit(hands.right_target(3, b), 3, b)
    assert not hands.right_hit(hands.right_target(0, b), 3, b)


def test_even_the_shortest_arms_reach_the_first_fret():
    import dataclasses
    from body.cppn import LIMITS
    b = generate_body(random_genome(np.random.default_rng(2)), 0)
    worst = dataclasses.replace(b, upper_arm=LIMITS["upper_arm"][0], forearm=LIMITS["forearm"][0],
                                torso=LIMITS["torso"][1], thigh=LIMITS["thigh"][1], shin=LIMITS["shin"][1],
                                neck=LIMITS["neck"][1], head_r=LIMITS["head_r"][1])
    for s in range(guitar.STRINGS):
        t = hands.left_target(s, 1, worst)
        sh, el = hands.reach(worst, hands.SITTING[0], t)
        assert math.dist(hands.hand(worst, hands.SITTING[0], sh, el), t) < 0.01
