import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from world import cppn, interest
from world.generate import KINDS, World, generate


@pytest.fixture(scope="module")
def genome():
    return cppn.random_genome(np.random.default_rng(3))


def test_features_are_periodic_over_the_world_width():
    x = np.array([0.0, 160.0])
    f = cppn.features(x, 160.0)
    np.testing.assert_allclose(f[0], f[1], atol=1e-9)
    assert f.shape == (2, 2 * len(cppn.WAVELENGTHS))


def test_the_same_seed_gives_the_same_world(genome):
    a, b = generate(genome, 17, 120.0), generate(genome, 17, 120.0)
    np.testing.assert_array_equal(a.heights, b.heights)
    assert a.palette == b.palette and a.name == b.name
    assert [(e.kind, e.x) for e in a.entities] == [(e.kind, e.x) for e in b.entities]


def test_different_seeds_give_different_worlds(genome):
    worlds = [generate(genome, s, 120.0) for s in range(6)]
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            assert np.abs(worlds[i].heights - worlds[j].heights).mean() > 0.05
            assert worlds[i].palette != worlds[j].palette


def test_world_is_a_ring_without_a_seam(genome):
    w = generate(genome, 5, 80.0)
    typical = np.abs(np.diff(w.heights)).max()
    assert abs(w.heights[-1] - w.heights[0]) <= typical + 1e-9
    assert w.height_at(0.3) == pytest.approx(w.height_at(80.3))
    assert w.height_at(-2.0) == pytest.approx(w.height_at(78.0))


def test_height_between_samples_is_interpolated(genome):
    w = generate(genome, 2, 80.0)
    i = 10
    mid = w.height_at((i + 0.5) * w.dx)
    assert mid == pytest.approx((w.heights[i] + w.heights[i + 1]) / 2)
    xs = np.array([0.1, 3.7, 79.9, 200.2])
    np.testing.assert_allclose(w.heights_at(xs), [w.height_at(x) for x in xs])


def test_entities_stand_on_the_ground_inside_the_world(genome):
    for seed in range(8):
        w = generate(genome, seed, 160.0)
        for e in w.entities:
            assert 0 <= e.x < w.width
            assert e.y == pytest.approx(w.height_at(e.x))
            assert e.kind in KINDS
            assert 0.4 <= e.size <= 1.6
            assert all(0 <= c <= 1 for c in e.color)


def test_colors_and_heights_stay_in_protective_limits(genome):
    for seed in range(10):
        w = generate(genome, seed, 120.0)
        for name, rgb in w.palette.items():
            assert len(rgb) == 3 and all(0 <= c <= 1 for c in rgb), name
        assert np.abs(w.heights).max() <= 6.0
        assert w.name and w.name[0].isupper()


def test_world_size_keeps_the_scale_of_hills(genome):
    small, big = generate(genome, 4, 80.0), generate(genome, 4, 320.0)
    assert len(big.heights) == 4 * len(small.heights)
    # число холмов растёт вместе с шириной, а крутизна та же
    slope = lambda w: np.abs(np.diff(w.heights)).mean()  # noqa: E731
    assert slope(big) == pytest.approx(slope(small), rel=0.5)


def flat_world(heights, sky=(0.6, 0.8, 1.0), ground=(0.3, 0.5, 0.2)):
    w = World.__new__(World)
    w.width, w.dx = len(heights) * 0.25, 0.25
    w.heights = np.asarray(heights, float)
    w.palette = {"sky_top": sky, "sky_bottom": sky, "ground": ground, "ground_deep": ground, "grass": ground}
    w.entities = []
    return w


def test_interest_prefers_hills_to_a_flat_plain():
    flat = flat_world(np.zeros(480))
    x = np.arange(480) * 0.25
    hills = flat_world(2.0 * np.sin(2 * np.pi * x / 20))
    assert interest.score(hills)["relief"] > interest.score(flat)["relief"]
    assert interest.score(flat)["passable"] == 1.0


def test_interest_punishes_walls_and_low_contrast():
    x = np.arange(480) * 0.25
    walls = flat_world(np.where((x // 6) % 2 == 0, 0.0, 3.0))
    assert interest.score(walls)["passable"] < 0.5
    grey = flat_world(np.zeros(480), sky=(0.5, 0.5, 0.5), ground=(0.5, 0.5, 0.5))
    assert interest.score(grey)["contrast"] == 0.0


@given(st.integers(min_value=0, max_value=2 ** 31 - 1))
def test_any_seed_gives_a_valid_world(seed):
    w = generate(cppn.random_genome(np.random.default_rng(1)), seed, 40.0)
    assert np.isfinite(w.heights).all()
    assert 0.0 <= interest.score(w)["total"] <= 1.0
