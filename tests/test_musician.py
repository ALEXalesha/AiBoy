import numpy as np
import pytest

from music import guitar, musician, songbook
from music.musician import IN, Musician, notes_of, step_input


@pytest.fixture
def phrase():
    return songbook.phrases(songbook.by_name("Чижик-пыжик"))[0]


def test_a_take_is_notes_on_the_guitar(phrase):
    m = Musician(seed=1)
    take = m.play(phrase.steps, phrase.meter, phrase.notes)
    assert take.inputs.shape == (phrase.steps, IN)
    for t, s, f, force, dur in notes_of(take.actions, take.steps):
        assert 0 <= s < guitar.STRINGS and 0 <= f < guitar.FRETS and 0 <= force < guitar.FORCES
        assert 1 <= dur and t + dur <= take.steps


def test_note_lasts_until_damped_or_restruck():
    a = np.zeros((8, 4), np.int64)
    a[0] = (1, musician.pos_of(2, 3), 1, 0)
    a[3] = (1, musician.pos_of(2, 5), 1, 0)     # та же струна - первая нота кончается
    a[5] = (0, 0, 0, 1)                         # глушение
    assert notes_of(a, 8) == [(0, 2, 3, 1, 3), (3, 2, 5, 1, 2)]


def test_target_is_described_per_string_and_free_play_is_zeros(phrase):
    a = np.zeros((phrase.steps, 4), np.int64)
    x = step_input(a, 0, phrase.steps, phrase.meter, phrase.notes)
    first = phrase.notes[0]
    base = musician.HISTORY_STEPS * musician.PER_STEP + 9
    assert x[base] == 1.0 and x[base + 1] == 0.0
    for s, o in enumerate(guitar.OPEN):
        assert x[base + 2 + s] == pytest.approx(np.clip((first.pitch - o) / 12.0, -1, 2))
    free = step_input(a, 0, 32, 8, None)
    assert not free[base:base + musician.AHEAD * musician.PER_TARGET].any() and free[-1] == 0.0
    now = musician.now_input(0, phrase.notes)
    marked = {musician.string_fret(i) for i in np.flatnonzero(now[:-1])}
    assert marked == set(guitar.positions(first.pitch)) and now[-1] == 1.0
    assert not musician.now_input(1, phrase.notes).any()


def test_reinforce_gradient_matches_numeric():
    m = Musician(seed=2, hidden=5, dtype=np.float64)
    m.skip.W[...] = np.random.default_rng(5).normal(0, 0.3, m.skip.W.shape)
    target = songbook.phrases(songbook.by_name("Чижик-пыжик"))[0].notes
    take = m.play(6, 8, target, rng=np.random.default_rng(0))
    take.actions[1, 0] = 1                                  # хоть один удар
    loss, grad = m._loss_grad(take, advantage=0.7)
    m.backward(grad)
    for name, p, g in m.params():
        g = g.copy()
        idx = tuple(np.random.default_rng(3).integers(0, s) for s in p.shape)
        old = p[idx]
        p[idx] = old + 1e-6
        up, _ = m._loss_grad(take, 0.7)
        p[idx] = old - 1e-6
        down, _ = m._loss_grad(take, 0.7)
        p[idx] = old
        assert g[idx] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-8), name


def test_positive_advantage_makes_the_choice_more_likely(phrase):
    m = Musician(seed=3)
    take = m.play(phrase.steps, phrase.meter, phrase.notes)
    before, _ = m._loss_grad(take, 1.0, entropy=0.0)
    m.baseline["x"] = 0.0
    for _ in range(5):
        m.learn(take, 1.0, key="x")
        m.baseline["x"] = 0.0
    after, _ = m._loss_grad(take, 1.0, entropy=0.0)
    assert after < before                                   # -Σ log π(a) уменьшилась


def test_save_and_load(tmp_path, phrase):
    m = Musician(seed=4)
    take = m.play(phrase.steps, phrase.meter, phrase.notes)
    m.learn(take, 0.5, key="Чижик-пыжик/0")
    m.save(tmp_path / "m.npz")
    other = Musician(seed=9).load(tmp_path / "m.npz")
    assert other.baseline == m.baseline and other.attempts == 1
    g1 = m.play(phrase.steps, phrase.meter, phrase.notes, greedy=True)
    g2 = other.play(phrase.steps, phrase.meter, phrase.notes, greedy=True)
    np.testing.assert_array_equal(g1.actions, g2.actions)


def test_he_learns_the_tune_noticeably(tmp_path):
    """Закон обучения (замер - в плане): зерно 0, «Чижик-пыжик», фраза 1, 200 попыток без
    владельца. Медиана похожести последних 20 выше медианы первых 20 не меньше чем на 0.2
    (на замере 0.14 -> 0.50; на зёрнах 1-3 прирост 0.21-0.31)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import measure_learning as ml
    closeness, _ = ml.run(attempts=200, seed=0, log=lambda *a: None)
    first, last = np.median(closeness[:20]), np.median(closeness[-20:])
    assert last > first + 0.2, (first, last)


def test_an_owner_rating_pushes_hard_but_does_not_blow_up_the_spread(phrase):
    """Оценка владельца - +2 к награде, это десятки обычных разбросов. Преимущество обрезано
    до ADV_CLIP разбросов, и разброс растёт не больше, чем от такого отклонения: иначе
    обычный сигнал похожести потом долго был бы в разы слабее."""
    m = Musician(seed=0)
    take = m.play(phrase.steps, phrase.meter, phrase.notes)
    m.baseline["k"], m.spread["k"] = 0.3, 0.1 ** 2
    used = m.learn(take, 0.3 + 2.0, key="k")
    assert used == pytest.approx(musician.ADV_CLIP, rel=0.02)
    assert m.spread["k"] <= 0.1 ** 2 * (1 + musician.BASELINE_K * (musician.ADV_CLIP ** 2 * 1.03 - 1))
    assert m.baseline["k"] < 0.3 + 0.1 * musician.ADV_CLIP * musician.BASELINE_K * 1.03
