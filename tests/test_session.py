import numpy as np
import pytest

from body.cppn import generate_body, random_genome
from music import session, songbook
from music.journal import Journal
from music.musician import Musician
from music.session import FREE_EVERY, LEARNED, Guitarist
from music.songbook import Songbook, Tune


@pytest.fixture
def body():
    return generate_body(random_genome(np.random.default_rng(2)), 3)


def make(tmp_path, mode="musician"):
    return Guitarist(Musician(seed=1), Journal(tmp_path / "j.json"), Songbook(tmp_path / "s.json"), mode=mode,
                     rng=np.random.default_rng(0))


def test_every_fourth_attempt_is_free_play_and_the_rest_learn_the_first_tune(tmp_path):
    g = make(tmp_path)
    targets = [g.target() for _ in range(8)]
    for i, (tune, phrase) in enumerate(targets, start=1):
        if i % FREE_EVERY == 0:
            assert tune is None and phrase is None
        else:
            assert tune == songbook.BUILTIN[0].name and phrase.index == 0


def test_the_owners_new_tune_comes_first_and_learned_phrases_are_skipped(tmp_path):
    g = make(tmp_path)
    g.book.add(Tune("Моя", ((60, 2), (64, 2), (67, 4)) * 6, 100, 8, builtin=False))
    assert g.target()[0] == "Моя"
    g.book.own = []
    first = songbook.BUILTIN[0].name
    g.journal.add({"tune": first, "phrase": 0, "notes": [], "A": LEARNED + 0.01, "B": 0.0, "C": None,
                   "taste": None, "total": 0.0, "rating": None})
    tune, phrase = g.target()
    assert tune == first and phrase.index == 1


def test_an_attempt_in_musician_mode_sounds_every_chosen_note(tmp_path, body):
    g = make(tmp_path)
    a = g.attempt(body)
    assert a.sounded == [1] * len(a.intended)
    assert len(a.notes()) == len(a.intended)
    assert a.arms.shape == (int(np.ceil(a.duration * session.FPS)) + 1, 4) and np.isfinite(a.arms).all()
    assert a.duration == pytest.approx(a.steps * 60 / a.tempo / 2)


def test_finish_scores_learns_and_writes_the_journal(tmp_path, body):
    g = make(tmp_path)
    a = g.attempt(body)
    entry, progress, novelty = g.finish(a, body)
    assert g.musician.attempts == 1 and len(g.journal.entries) == 1
    assert entry["tune"] == a.tune and entry["A"] is not None and -1 <= entry["B"] <= 1
    assert entry["C"] is None and entry["rating"] is None
    assert 0.0 <= progress <= 1.0 and novelty == pytest.approx(1.0)
    assert len(entry["notes"]) == len(a.intended)


def test_owner_rating_after_the_attempt_counts_twice_as_heavy_as_anything(tmp_path, body):
    g = make(tmp_path)
    a = g.attempt(body)
    entry, _, _ = g.finish(a, body)
    before_total = entry["total"]
    g.rate_last(1)
    assert g.journal.get(entry["id"])["rating"] == 1
    assert entry["total"] == pytest.approx(before_total + 2.0)
    assert len(g.taste.ratings) == 1 and g.musician.attempts == 2


def test_rating_given_during_the_attempt_goes_into_the_reward(tmp_path, body):
    g = make(tmp_path)
    a = g.attempt(body)
    entry, _, _ = g.finish(a, body, rating=-1)
    assert entry["C"] == -1.0 and entry["rating"] == -1
    assert len(g.taste.ratings) == 1


def test_arm_track_follows_commands_smoothly():
    b = generate_body(random_genome(np.random.default_rng(2)), 0)
    track = session.arm_track(b, [(0, (1.0, 0.5, 0.5, 1.5))], 90)
    assert np.abs(np.diff(track, axis=0)).max() < 0.12
    assert track[-1] == pytest.approx([1.0, 0.5, 0.5, 1.5], abs=0.02)
