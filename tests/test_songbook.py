import json

import pytest

from music import guitar, songbook
from music.songbook import Songbook, Tune


def test_ten_builtin_public_domain_tunes():
    tunes = songbook.BUILTIN
    assert len(tunes) == 10
    assert len({t.name for t in tunes}) == 10
    assert not any("узнечик" in t.name for t in tunes)          # мелодия Шаинского - под охраной
    for t in tunes:
        steps = sum(d for _, d in t.notes)
        measures = steps / t.meter
        assert 8 <= measures <= 17, (t.name, measures)          # 8-16 тактов (+ затакт)
        assert 70 <= t.tempo <= 120
        assert all(d >= 1 for _, d in t.notes)


def test_tunes_fit_the_guitar_after_octave_shift():
    for t in songbook.BUILTIN:
        pitches = [p for p, _ in songbook.in_range(t).notes if p is not None]
        assert guitar.LOW <= min(pitches) and max(pitches) <= guitar.HIGH, t.name
        original = [p for p, _ in t.notes if p is not None]
        shift = pitches[0] - original[0]
        assert shift % 12 == 0 and all(a - b == shift for a, b in zip(pitches, original))


def test_phrases_are_four_measures_with_onsets_from_zero():
    t = songbook.by_name("Чижик-пыжик")
    ph = songbook.phrases(t)
    assert len(ph) == 2
    for p in ph:
        assert p.notes[0].onset == 0
        assert p.steps == 4 * t.meter
        assert all(0 <= n.onset < p.steps and n.onset + n.dur <= p.steps for n in p.notes)
    first = [n.pitch for n in ph[0].notes]
    assert first[:7] == [64, 60, 64, 60, 65, 64, 62]                # ми до ми до фа ми ре


def test_rests_are_not_notes_but_keep_time():
    t = Tune("с паузой", ((60, 2), (None, 2), (62, 4)), 100, 8)
    notes = songbook.phrases(t)[0].notes
    assert [(n.onset, n.pitch, n.dur) for n in notes] == [(0, 60, 2), (4, 62, 4)]


def test_own_tunes_are_saved_next_to_builtin_ones(tmp_path):
    path = tmp_path / "songbook.json"
    book = Songbook.load(path)
    assert len(book.all()) == 10
    book.add(Tune("Моя", ((60, 2), (64, 2), (67, 4)), 90, 8, builtin=False))
    assert book.save()
    again = Songbook.load(path)
    assert [t.name for t in again.all()][-1] == "Моя"
    assert again.by_name("Моя").notes == ((60, 2), (64, 2), (67, 4))


@pytest.mark.parametrize("junk", ["{обрыв", "[1, 2]", json.dumps([{"name": "x", "notes": "abc"}]),
                                  json.dumps([{"name": "", "notes": [[60, 2]]}])])
def test_broken_songbook_file_keeps_builtin_tunes(tmp_path, junk):
    path = tmp_path / "songbook.json"
    path.write_text(junk, encoding="utf-8")
    assert len(Songbook.load(path).all()) == 10


def test_own_tune_cannot_replace_a_builtin_name(tmp_path):
    book = Songbook.load(tmp_path / "s.json")
    with pytest.raises(ValueError):
        book.add(Tune("Чижик-пыжик", ((60, 2),), 100, 8, builtin=False))
