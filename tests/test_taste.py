import numpy as np
import pytest

from music import taste
from music.reward import Played
from music.taste import Taste


def random_take(rng):
    n = int(rng.integers(6, 20))
    base = int(rng.integers(42, 70))
    return [Played(int(t), int(np.clip(base + rng.integers(-5, 6), 40, 76)), int(rng.integers(1, 4)),
                   int(rng.integers(0, 3))) for t in sorted(rng.choice(32, n, replace=False))]


def likes_high(notes):
    return float(np.mean([n.pitch for n in notes])) > 56


def test_it_guesses_a_hidden_rule_after_50_ratings():
    """Скрытое правило владельца - «нравятся высокие ноты». После 50 оценок модель
    угадывает оценку новых попыток не реже чем в 80 % случаев."""
    rng = np.random.default_rng(0)
    t = Taste()
    for _ in range(50):
        notes = random_take(rng)
        t.rate(taste.features(notes, None), 1 if likes_high(notes) else -1)
    tests = [random_take(rng) for _ in range(200)]
    hits = [(t.predict(taste.features(n, None)) > 0) == likes_high(n) for n in tests]
    assert np.mean(hits) >= 0.8


def test_without_ratings_it_has_no_opinion():
    notes = random_take(np.random.default_rng(1))
    assert Taste().predict(taste.features(notes, None)) == 0.0


def test_accuracy_counts_guesses_made_before_the_rating():
    rng = np.random.default_rng(2)
    t = Taste()
    assert t.accuracy() is None
    for _ in range(40):
        notes = random_take(rng)
        t.rate(taste.features(notes, None), 1 if likes_high(notes) else -1)
    acc = t.accuracy()
    assert acc is not None and 0.5 <= acc <= 1.0
    assert t.guesses == 39                                  # первая оценка - без догадки


def test_rebuilt_from_saved_ratings_gives_the_same_guess():
    rng = np.random.default_rng(3)
    t = Taste()
    for _ in range(20):
        notes = random_take(rng)
        t.rate(taste.features(notes, None), 1 if likes_high(notes) else -1)
    again = Taste.from_ratings(t.ratings)
    probe = taste.features(random_take(rng), None)
    assert again.predict(probe) == pytest.approx(t.predict(probe), abs=1e-6)


def test_features_are_bounded_and_fixed_size():
    rng = np.random.default_rng(4)
    for _ in range(50):
        f = taste.features(random_take(rng), 0.4)
        assert f.shape == (taste.FEATURES,) and np.isfinite(f).all() and np.abs(f).max() <= 3
    assert taste.features([], None).shape == (taste.FEATURES,)


def test_trust_grows_with_consistent_ratings_and_stays_near_zero_for_random_ones():
    """Догадка вкуса входит в награду с доверием: без оценок - 0; на согласных оценках растёт;
    на случайных (вкус угадывает как монетка) - остаётся около 0, чтобы шум не увёл музыканта."""
    assert Taste().trust() == 0.0
    rng = np.random.default_rng(4)
    good, noisy = Taste(), Taste()
    trust_good = []
    for i in range(60):
        notes = random_take(rng)
        good.rate(taste.features(notes, None), 1 if likes_high(notes) else -1)
        noisy.rate(taste.features(notes, None), 1 if rng.random() < 0.5 else -1)
        trust_good.append(good.trust())
    assert trust_good[3] == 0.0 and 0.0 < trust_good[10] < trust_good[-1] <= 1.0
    assert good.trust() > 0.5
    assert noisy.trust() < 0.2


def test_it_learns_a_taste_for_closeness_to_the_tune():
    """Владельцу нравится, когда похоже на мелодию (A > 0.5), а ноты сами по себе ему
    безразличны: вкус должен опереться на похожесть."""
    rng = np.random.default_rng(5)
    t = Taste()
    for _ in range(40):
        a = float(rng.random())
        t.rate(taste.features(random_take(rng), a), 1 if a > 0.5 else -1)
    tests = [(random_take(rng), float(rng.random())) for _ in range(200)]
    hits = [(t.predict(taste.features(n, a)) > 0) == (a > 0.5) for n, a in tests if abs(a - 0.5) > 0.1]
    assert np.mean(hits) >= 0.85
