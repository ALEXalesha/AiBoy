import numpy as np
import pytest

from body.cppn import generate_body, random_genome as body_genome
from body.physics import Human
from brain import observe, state
from brain.brain import Brain, run_steps
from world import cppn
from world.generate import generate


class Ground:
    def __init__(self, fn, width=200.0, entities=()):
        self.fn, self.width, self.entities, self.night = fn, width, list(entities), False

    def height_at(self, x):
        return float(self.fn(x % self.width))


@pytest.fixture(scope="module")
def body():
    return generate_body(body_genome(np.random.default_rng(2)), 5)


@pytest.fixture(scope="module")
def wgenome():
    return cppn.random_genome(np.random.default_rng(3))


def test_observation_has_fixed_size_and_stays_bounded(body, wgenome):
    w = generate(wgenome, 1, 80.0)
    h = Human(body, w, x=10.0)
    o = observe.observe(h, w)
    assert o.shape == (observe.OBS_DIM,) and o.dtype == np.float32
    assert np.isfinite(o).all() and np.abs(o).max() <= 3.0


def test_observation_is_egocentric(body):
    """Холм справа: смотрит вправо - холм впереди, влево - позади (зеркально)."""
    hill = Ground(lambda x: 2.0 if 12.5 < x < 15 else 0.0)
    h = Human(body, hill, x=10.0)
    right = observe.observe(h, hill)[observe.TERRAIN]
    h.facing = -1
    left = observe.observe(h, hill)[observe.TERRAIN]
    assert right[-1] > 0.5 and right[0] == 0
    np.testing.assert_allclose(right, left[::-1], atol=1e-6)


def test_state_sees_a_hill_on_the_right_and_a_pit_on_the_left(body):
    world = Ground(lambda x: 2.5 if 14 < x < 17 else (-2.5 if 3 < x < 6 else 0.0))
    h = Human(body, world, x=10.0)
    s = state.summarize(h, world, want_dir=0.0, want_jump=0.0, curiosity=0.5, fell=0.0)
    f = dict(zip(state.FIELDS, s))
    assert len(s) == state.STATE_DIM == len(state.FIELDS)
    assert f["hill_right"] > 0.5 and f["hill_left"] < 0.1
    assert f["pit_left"] > 0.5 and f["pit_right"] < 0.1
    assert f["flat"] < 0.5


def test_state_names_the_nearest_entity(body):
    from world.generate import Entity
    ent = Entity("mushroom", 12.0, 0.0, 1.0, (0.9, 0.1, 0.1), 0.5)
    world = Ground(lambda x: 0.0, entities=[ent])
    h = Human(body, world, x=10.0)
    f = dict(zip(state.FIELDS, state.summarize(h, world, 0.0, 0.0, 0.0, 0.0)))
    assert f["entity"] == 1.0 and f["kind_mushroom"] == 1.0 and f["entity_side"] > 0
    assert f["entity_dist"] < 0.3


def life(brain, world, body, steps, learn=True, x=5.0):
    h = Human(body, world, x=x)
    log = run_steps(brain, h, world, steps, learn=learn)
    return h, log


def test_predictor_error_falls_while_it_learns(body, wgenome):
    brain = Brain(seed=1)
    _, log = life(brain, generate(wgenome, 2, 80.0), body, 1500)
    err = np.array(log["pred_err"])
    assert err[-200:].mean() < 0.7 * err[20:220].mean()


def test_human_moves_from_his_place_in_part_of_the_worlds(body, wgenome):
    moved = 0
    for seed in range(6):
        w = generate(wgenome, seed, 80.0)
        h, _ = life(Brain(seed=seed), w, body, 900)
        moved += h.odometer > 1.0
    assert moved >= 2


def test_long_life_does_not_produce_nan(body, wgenome):
    brain = Brain(seed=4)
    h, log = life(brain, generate(wgenome, 7, 80.0), body, 3000)
    for _, p, _ in brain.params():
        assert np.isfinite(p).all()
    assert np.isfinite([h.px, h.py]).all() and np.isfinite(log["reward"]).all()


def test_talk_heads_are_not_trained(body, wgenome):
    brain = Brain(seed=5)
    before = [p.copy() for _, p, _ in brain.talk.params()]
    trunk = brain.cell.Wx.copy()
    life(brain, generate(wgenome, 3, 80.0), body, 400)
    for b, (_, p, _) in zip(before, brain.talk.params()):
        np.testing.assert_array_equal(b, p)
    assert not np.array_equal(trunk, brain.cell.Wx)


def test_without_learning_weights_stay_the_same(body, wgenome):
    brain = Brain(seed=6)
    before = [p.copy() for _, p, _ in brain.params()]
    life(brain, generate(wgenome, 3, 80.0), body, 200, learn=False)
    for b, (_, p, _) in zip(before, brain.params()):
        np.testing.assert_array_equal(b, p)


def test_save_and_load_give_the_same_decisions(tmp_path, body, wgenome):
    brain = Brain(seed=7)
    w = generate(wgenome, 3, 80.0)
    life(brain, w, body, 300)
    path = tmp_path / "brain.npz"
    brain.save(path)
    other = Brain(seed=99)
    other.load(path)
    assert other.steps == brain.steps
    h = Human(body, w, x=5.0)
    o = observe.observe(h, w)
    a = brain.decide(o, explore=False)
    b = other.decide(o, explore=False)
    np.testing.assert_allclose(a.targets, b.targets, rtol=1e-6)


def test_loading_a_broken_file_raises(tmp_path):
    bad = tmp_path / "brain.npz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        Brain(seed=0).load(bad)
