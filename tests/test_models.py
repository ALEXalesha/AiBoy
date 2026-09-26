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
    for name in ("world", "body", "speech", "brain"):
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
    # флаги гитары в игре - да или нет, и играет он в пятой части жизни
    playing = rng.random(n) < 0.2
    s[:, IDX["playing"]] = playing
    for k in ("learning_tune", "got_it", "failing"):
        s[:, IDX[k]] = playing & (rng.random(n) < 0.4)
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


def test_starter_brain_passport_is_honest(m):
    brain = models.starter_brain()
    pp = models.passport("brain")["honest_training_time"]
    assert brain.steps == pp["brain_steps"] > 0
    assert pp["lived_hours_online"] == pytest.approx(brain.steps / 30 / 3600, abs=0.01)
    assert pp["es_generations"] > 0 and pp["es_minutes"] > 0


def test_starter_brain_moves_in_part_of_the_worlds(m):
    moved = 0
    for i in range(5):
        w = generate(m.world_genome, 8_000 + i, 160.0)
        h = Human(generate_body(m.body_genome, 8_100 + i), w, x=80.0)
        brain = models.starter_brain()
        run_steps(brain, h, w, 900, learn=False)
        moved += abs(h.px - 80.0) > 1.0
    assert moved >= 2


def test_no_world_has_a_trap(m):
    """Закон мира: из любой точки можно попасть во весь мир шагом или прыжком - на 60 новых
    зёрнах, ни одной ямы-ловушки."""
    from world import reach
    for seed in range(20_000, 20_060):
        w = generate(m.world_genome, seed, 160.0)
        assert reach.traps(w) == [], seed


# Пороги законов поведения - по замеру стартового мозга при сборке (README, «Замеры»):
# первая сборка разворачивалась 577 раз в минуту, суставы дёргались назад в 15 % кадров,
# средний |изменение угла| - 0.082 рад за кадр.
CALM = {"turns_per_min": 10.0, "jitter": 0.08, "dangle": 0.04}
MOVES_M_PER_MIN = 12.0


@pytest.fixture(scope="module")
def behaviour(m):
    from game import probe
    return probe.summary(m, models.starter_brain, [(30_000 + i, 31_000 + i) for i in range(6)], 60, learn=True)


def test_starter_brain_is_calm_in_the_game(behaviour):
    s, rows = behaviour
    for key, limit in CALM.items():
        assert s[key] <= limit, (key, s[key], [r[key] for r in rows])


def test_starter_brain_still_walks_somewhere(behaviour):
    s, rows = behaviour
    assert s["displacement_m"] >= MOVES_M_PER_MIN, [round(r["displacement_m"], 1) for r in rows]
    assert s["moved_over_2m"] >= 5


def test_starter_brain_climbs_out_of_the_worst_hollow(m):
    """Со дна самой глубокой впадины мира выбирается за 30 с игры - в 90 % миров и больше."""
    from game import probe
    got = [probe.escape(m, models.starter_brain, 40_000 + i, 41_000 + i, 30)[0] for i in range(20)]
    assert sum(got) >= 18, got
