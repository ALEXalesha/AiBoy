"""Обученные модели из models/: то, что проверяли при сборке, держится и сейчас."""
import numpy as np
import pytest

from body.cppn import generate_body
from body.physics import Human
from brain.brain import run_steps
from brain.state import IDX, STATE_DIM
from game import models
from teacher import phrases
from world import interest
from world.generate import generate


@pytest.fixture(scope="module")
def m():
    return models.load()


def test_every_model_is_trained_and_has_a_passport(m):
    assert all(m.trained.values()), m.trained
    for name in ("world", "body", "speech", "voice", "brain"):
        assert models.passport(name), name


def test_evolved_world_is_passable_on_most_of_the_way(m):
    """Перепады в пределах прыжка на большей части мира - на 30 новых зёрнах."""
    for seed in range(30):
        w = generate(m.world_genome, 7_000 + seed, 160.0)
        assert interest.passable_fraction(w) > 0.75, seed
        assert interest.walls_per_100m(w) < 4.0, seed


def test_evolved_worlds_are_more_interesting_than_random_ones(m):
    rng = np.random.default_rng(0)
    from world import cppn
    evolved = np.mean([interest.score(generate(m.world_genome, s, 160.0))["total"] for s in range(20)])
    rand = np.mean([interest.score(generate(cppn.random_genome(rng, 0.6), s, 160.0))["total"] for s in range(20)])
    assert evolved > rand + 0.05
    assert evolved == pytest.approx(models.passport("world")["validation_new_seeds"]["interest_mean"], abs=0.08)


def test_worlds_of_the_evolved_net_differ(m):
    worlds = [generate(m.world_genome, s, 160.0) for s in range(12)]
    assert interest.diversity(worlds) > 0.3
    assert len({w.name for w in worlds}) >= 10
    assert any(w.night for w in generate_many(m)) and not all(w.night for w in generate_many(m))


def generate_many(m):
    return [generate(m.world_genome, s, 80.0) for s in range(40)]


def random_states(n, rng):
    s = rng.random((n, STATE_DIM)).astype(np.float32)
    from brain.state import FIELDS
    kinds = [i for i, f in enumerate(FIELDS) if f.startswith("kind_")]
    s[:, kinds] = 0
    s[np.arange(n), rng.choice(kinds, n)] = 1
    s[:, IDX["entity_side"]] = rng.choice([-1.0, 1.0], n)
    s[:, IDX["want_dir"]] = rng.uniform(-1, 1, n)
    s[:, IDX["fell"]] = (rng.random(n) < 0.15).astype(np.float32)
    s[:, IDX["facing"]] = rng.choice([-1.0, 1.0], n)
    return s


def test_speech_uses_teacher_words_in_most_phrases(m):
    rng = np.random.default_rng(3)
    words = phrases.vocabulary()
    samples = [m.speech.sample(s, rng, 0.75) for s in random_states(150, rng)]
    good = [bool(p) and set(phrases.words(p)) <= words for p in samples]
    assert np.mean(good) > 0.8, [p for p, g in zip(samples, good) if not g][:10]
    assert len(set(samples)) > 40


def test_speech_differs_for_different_states(m):
    def st(**kw):
        s = np.zeros(STATE_DIM, np.float32)
        s[IDX["grounded"]] = 1
        s[IDX["facing"]] = 1
        s[IDX["curiosity"]] = 0.4
        for k, v in kw.items():
            s[IDX[k]] = v
        return s
    rng = np.random.default_rng(4)
    fall = [m.speech.sample(st(fell=1.0), rng, 0.75) for _ in range(30)]
    hill = [m.speech.sample(st(hill_right=1.0), rng, 0.75) for _ in range(30)]
    assert sum(p in phrases.FALL for p in fall) >= 12
    assert not set(fall) & set(hill) or len(set(fall) & set(hill)) < 5
    assert sum("справа" in p or "повыше" in p for p in hill) >= 5


def test_voice_follows_its_teacher(m):
    """На состояниях из настоящей жизни (как при обучении, но других миров)."""
    from teacher import voice as tv
    from train.speech import collect_states
    rng = np.random.default_rng(5)
    s = collect_states(300, seed=9, log=lambda *a: None)
    err = np.mean([(m.voice.params_for(x) - tv.target(x, rng, noise=0.0)) ** 2 for x in s])
    assert err < 0.02


def test_starter_brain_passport_is_honest(m):
    brain = models.starter_brain()
    pp = models.passport("brain")["honest_training_time"]
    assert brain.steps == pp["brain_steps"] > 0
    assert pp["lived_hours"] == pytest.approx(brain.steps / 30 / 3600, abs=0.01)


def test_starter_brain_moves_in_part_of_the_worlds(m):
    moved = 0
    for i in range(5):
        w = generate(m.world_genome, 8_000 + i, 160.0)
        h = Human(generate_body(m.body_genome, 8_100 + i), w, x=80.0)
        brain = models.starter_brain()
        run_steps(brain, h, w, 900, learn=False)
        moved += abs(h.px - 80.0) > 1.0
    assert moved >= 2
