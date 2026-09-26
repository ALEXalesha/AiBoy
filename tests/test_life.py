import numpy as np
import pytest

from brain.brain import Brain
from game import models
from game.life import PHRASE_PAUSE, Life


@pytest.fixture(scope="module")
def m():
    return models.load()


def make(m, world=3, body=4, freq="normal", seed=0):
    return Life(m, world, body, 80.0, Brain(seed=seed), frequency=freq, seed=seed)


def run(life, ticks):
    events = []
    for _ in range(ticks):
        events += life.tick()
    return events


def test_three_hundred_steps_of_life(m):
    life = make(m)
    run(life, 300)
    assert life.time == pytest.approx(300 / 60)
    assert np.isfinite([life.human.px, life.human.py]).all()
    assert life.brain.steps == 150


def test_the_same_seeds_give_the_same_world_and_human(m):
    a, b = make(m, 11, 12), make(m, 11, 12)
    assert a.world.name == b.world.name and a.body == b.body
    assert make(m, 11, 13).body != a.body


def test_phrases_come_and_respect_the_pause(m):
    life = make(m, freq="often", seed=2)
    times = []
    for _ in range(60 * 60):
        for kind, *data in life.tick():
            if kind == "phrase":
                times.append(life.time)
                assert isinstance(data[0], str) and data[0]
    assert len(times) >= 3
    assert min(np.diff(times)) >= PHRASE_PAUSE["often"] - 1e-6
    assert len(life.phrases) == len(times)


def test_rare_frequency_talks_less_than_often(m):
    counts = {}
    for freq in ("rare", "often"):
        life = make(m, freq=freq, seed=5)
        run(life, 60 * 60)
        counts[freq] = len(life.phrases)
    assert counts["rare"] < counts["often"]


def test_without_learning_the_brain_stays_the_same(m):
    life = make(m, seed=6)
    life.learn = False
    before = [p.copy() for _, p, _ in life.brain.params()]
    run(life, 200)
    for b, (_, p, _) in zip(before, life.brain.params()):
        np.testing.assert_array_equal(b, p)


def test_curiosity_is_logged_for_the_panel(m):
    life = make(m, seed=7)
    run(life, 120)
    assert len(life.curiosity_log) >= 3
    assert all(np.isfinite(v) for v in life.curiosity_log)
