import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from music import reward, songbook
from music.reward import Played


def calm_line(pitches, gap=2):
    return [Played(i * gap, p, gap, 1) for i, p in enumerate(pitches)]


def chord(pitches, at=0):
    return [Played(at, p, 4, 1) for p in pitches]


def test_consonant_chord_beats_a_dissonant_one():
    good = chord([48, 52, 55], 0) + chord([48, 52, 55], 8) + chord([53, 57, 60], 16) + chord([55, 59, 62], 24)
    bad = chord([48, 49, 50], 0) + chord([48, 49, 50], 8) + chord([53, 54, 55], 16) + chord([55, 56, 57], 24)
    assert reward.consonance(good) > reward.consonance(bad) + 0.3
    assert reward.pleasant(good, 8, []).total > reward.pleasant(bad, 8, []).total


def test_exact_repeat_of_the_tune_is_max_closeness():
    ph = songbook.phrases(songbook.by_name("Чижик-пыжик"))[0]
    played = [Played(n.onset, n.pitch, n.dur, 1) for n in ph.notes]
    assert reward.closeness(played, ph.notes) == pytest.approx(1.0)


def test_time_shift_lowers_closeness_smoothly():
    ph = songbook.phrases(songbook.by_name("Ода к радости"))[0]
    values = []
    for k in range(6):
        played = [Played(n.onset + k, n.pitch, n.dur, 1) for n in ph.notes]
        values.append(reward.closeness(played, ph.notes))
    assert values[0] == pytest.approx(1.0)
    diffs = -np.diff(values)
    assert (diffs > 0).all() and (diffs <= 0.25).all(), values


def test_wrong_notes_and_missing_notes_lower_closeness():
    ph = songbook.phrases(songbook.by_name("Mary Had a Little Lamb"))[0]
    exact = [Played(n.onset, n.pitch, n.dur, 1) for n in ph.notes]
    wrong = [Played(n.onset, n.pitch + 1, n.dur, 1) for n in ph.notes]
    half = exact[::2]
    assert reward.closeness(wrong, ph.notes) < 0.8
    assert reward.closeness(half, ph.notes) < 0.75
    assert reward.closeness([], ph.notes) == 0.0


def test_silence_and_mush_are_punished():
    calm = calm_line([48, 50, 52, 53, 55, 57, 59, 60] * 2)
    silent = [Played(0, 48, 2, 1)]
    mush = [Played(t, 40 + (t * 7 + s * 5) % 30, 1, 2) for t in range(32) for s in range(4)]
    b_calm = reward.pleasant(calm, 8, []).total
    assert reward.pleasant(silent, 8, []).total < b_calm - 0.3
    assert reward.pleasant(mush, 8, []).total < b_calm - 0.3


def test_novelty_falls_when_he_repeats_himself():
    line = calm_line([48, 52, 55, 60, 55, 52, 48, 47])
    other = calm_line([50, 53, 57, 62, 57, 53, 50, 45])
    assert reward.novelty(line, []) == pytest.approx(1.0)
    assert reward.novelty(line, [line]) == pytest.approx(0.0)
    assert 0.3 < reward.novelty(line, [other]) <= 1.0


def test_total_weights_the_owner_heaviest():
    assert reward.WEIGHTS["C"] > reward.WEIGHTS["A"] and reward.WEIGHTS["C"] > reward.WEIGHTS["B"]
    r = reward.total(a=0.5, b=0.2, c=1.0)
    assert r == pytest.approx(reward.WEIGHTS["A"] * 0.5 + reward.WEIGHTS["B"] * 0.2 + reward.WEIGHTS["C"])
    assert reward.total(a=None, b=0.2, c=None) == pytest.approx(reward.WEIGHTS["B"] * 0.2)


notes_st = st.lists(st.tuples(st.integers(0, 31), st.integers(40, 76), st.integers(1, 8), st.integers(0, 2)),
                    max_size=60)


@given(notes_st, notes_st)
def test_components_stay_in_range(a, b):
    pa = [Played(*x) for x in a]
    pb = [Played(*x) for x in b]
    ph = [songbook.Target(n.onset, n.pitch, n.dur) for n in pb]
    parts = reward.pleasant(pa, 8, [pb])
    for v in (parts.consonance, parts.rhythm, parts.novelty, parts.total):
        assert -1.0 <= v <= 1.0
    assert 0.0 <= reward.closeness(pa, ph) <= 1.0
