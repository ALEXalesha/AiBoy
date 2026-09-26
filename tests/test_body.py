import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from body import skeleton
from body.cppn import LIMITS, Body, generate_body, random_genome


@pytest.fixture(scope="module")
def genome():
    return random_genome(np.random.default_rng(2))


@given(st.integers(min_value=0, max_value=2 ** 31 - 1))
def test_proportions_stay_in_limits(seed):
    b = generate_body(random_genome(np.random.default_rng(4)), seed)
    for name, (lo, hi) in LIMITS.items():
        assert lo <= getattr(b, name) <= hi, name
    for rgb in (b.skin, b.hair, b.shirt, b.pants, b.shoes):
        assert all(0 <= c <= 1 for c in rgb)
    assert b.hair_style in range(4)
    assert b.name and b.name[0].isupper()


def test_the_same_seed_gives_the_same_human_and_different_seeds_differ(genome):
    assert generate_body(genome, 9) == generate_body(genome, 9)
    bodies = [generate_body(genome, s) for s in range(8)]
    assert len({(b.thigh, b.shirt) for b in bodies}) == 8


def test_body_height_is_human(genome):
    for s in range(20):
        b = generate_body(genome, s)
        assert 1.35 <= b.height() <= 2.0
        assert b.thigh + b.shin > b.torso * 0.6


def bone_pairs():
    return [("pelvis", "neck"), ("sh_l", "el_l"), ("el_l", "ha_l"), ("sh_r", "el_r"), ("el_r", "ha_r"),
            ("hip_l", "kn_l"), ("kn_l", "ft_l"), ("hip_r", "kn_r"), ("kn_r", "ft_r")]


def test_skeleton_has_15_connected_points(genome):
    b = generate_body(genome, 1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        angles = skeleton.random_angles(rng)
        pts = skeleton.points(b, angles, facing=1)
        assert set(pts) == set(skeleton.POINTS) and len(skeleton.POINTS) == 15
        lengths = {("pelvis", "neck"): b.torso, ("sh_l", "el_l"): b.upper_arm, ("el_l", "ha_l"): b.forearm,
                   ("sh_r", "el_r"): b.upper_arm, ("el_r", "ha_r"): b.forearm, ("hip_l", "kn_l"): b.thigh,
                   ("kn_l", "ft_l"): b.shin, ("hip_r", "kn_r"): b.thigh, ("kn_r", "ft_r"): b.shin}
        for (a, c), length in lengths.items():
            assert math.dist(pts[a], pts[c]) == pytest.approx(length), (a, c)
        head_gap = math.dist(pts["neck"], pts["head"])
        assert head_gap == pytest.approx(b.neck + b.head_r)


def test_left_and_right_limbs_are_symmetric(genome):
    b = generate_body(genome, 3)
    angles = skeleton.rest_angles()
    for side_a, side_b in (("sh_l", "sh_r"), ("el_l", "el_r"), ("hip_l", "hip_r"), ("kn_l", "kn_r")):
        angles[skeleton.JOINTS.index(side_a)] = 0.7
        angles[skeleton.JOINTS.index(side_b)] = 0.7
    pts = skeleton.points(b, angles, facing=1)
    for a, c in (("ha_l", "ha_r"), ("el_l", "el_r"), ("ft_l", "ft_r"), ("kn_l", "kn_r")):
        assert pts[a] == pytest.approx(pts[c])


def test_facing_left_mirrors_the_body(genome):
    b = generate_body(genome, 3)
    angles = skeleton.random_angles(np.random.default_rng(1))
    right = skeleton.points(b, angles, facing=1)
    left = skeleton.points(b, angles, facing=-1)
    for name in skeleton.POINTS:
        assert left[name][0] == pytest.approx(-right[name][0])
        assert left[name][1] == pytest.approx(right[name][1])


def test_hip_forward_moves_the_foot_forward_and_knee_bends_back(genome):
    b = generate_body(genome, 0)
    a = skeleton.rest_angles()
    a[:] = 0
    base = skeleton.points(b, a, 1)
    a[skeleton.JOINTS.index("hip_r")] = 0.6
    assert skeleton.points(b, a, 1)["ft_r"][0] > base["ft_r"][0]
    a[skeleton.JOINTS.index("hip_r")] = 0
    a[skeleton.JOINTS.index("kn_r")] = 1.0
    bent = skeleton.points(b, a, 1)
    assert bent["ft_r"][0] < base["ft_r"][0] and bent["ft_r"][1] > base["ft_r"][1]


def test_angles_are_clipped_to_joint_limits():
    wild = np.full(len(skeleton.JOINTS), 50.0)
    clipped = skeleton.clip(wild)
    for j, (lo, hi) in enumerate(skeleton.JOINT_LIMITS):
        assert lo <= clipped[j] <= hi


def test_body_is_a_dataclass_with_a_name():
    assert "name" in Body.__dataclass_fields__
