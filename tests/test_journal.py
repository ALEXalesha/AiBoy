import json

import pytest

from music.journal import KEEP_FIRST, KEEP_LAST, MAX_ENTRIES, Journal


def entry(tune="Чижик-пыжик", a=0.3):
    return {"tune": tune, "phrase": 0, "mode": "musician", "tempo": 110, "meter": 8, "steps": 32,
            "notes": [[0, 2, 2, 1, 2, 1], [2, 1, 3, 1, 2, 1]], "A": a, "B": 0.1, "C": None, "taste": 0.0,
            "total": 0.35, "rating": None}


def test_journal_survives_a_restart(tmp_path):
    path = tmp_path / "j.json"
    j = Journal.load(path)
    first = j.add(entry(a=0.1))
    j.add(entry(a=0.5))
    j.rate(first["id"], 1)
    assert j.save()
    again = Journal.load(path)
    assert [e["A"] for e in again.entries] == [0.1, 0.5]
    assert again.get(first["id"])["rating"] == 1
    assert again.add(entry())["id"] == 3                  # номера продолжаются


def test_thinning_keeps_first_every_tenth_and_last(tmp_path):
    j = Journal(tmp_path / "j.json")
    for _ in range(MAX_ENTRIES + 1):
        j.add(entry())
    ids = [e["id"] for e in j.entries]
    assert len(ids) <= MAX_ENTRIES
    assert ids[:KEEP_FIRST] == list(range(1, KEEP_FIRST + 1))
    assert ids[-KEEP_LAST:] == list(range(MAX_ENTRIES + 2 - KEEP_LAST, MAX_ENTRIES + 2))
    middle = [i for i in ids if KEEP_FIRST < i < MAX_ENTRIES + 2 - KEEP_LAST]
    assert middle and all(i % 10 == 0 for i in middle)
    assert middle == [i for i in range(KEEP_FIRST + 1, MAX_ENTRIES + 2 - KEEP_LAST) if i % 10 == 0]


def test_first_of_a_tune_for_the_comparison(tmp_path):
    j = Journal(tmp_path / "j.json")
    j.add(entry("Ода к радости"))
    j.add(entry(None))
    j.add(entry("Чижик-пыжик", 0.2))
    last = j.add(entry("Чижик-пыжик", 0.6))
    assert j.first_of("Чижик-пыжик")["id"] == 3
    assert j.first_of(None)["id"] == 2
    assert [a for _, a in j.progress("Чижик-пыжик")] == [0.2, 0.6]
    assert j.best("Чижик-пыжик") == 0.6 and last["id"] == 4


@pytest.mark.parametrize("junk", ["{обрыв", "[1,2]", json.dumps({"entries": [{"id": "x"}, 5]}),
                                  json.dumps({"entries": [{**entry(), "id": 1, "notes": "abc"}]})])
def test_broken_journal_starts_clean(tmp_path, junk):
    path = tmp_path / "j.json"
    path.write_text(junk, encoding="utf-8")
    j = Journal.load(path)
    assert j.entries == [] or all(isinstance(e["notes"], list) for e in j.entries)
    j.add(entry())
    assert j.save()


def test_ratings_rebuild_the_taste(tmp_path):
    j = Journal(tmp_path / "j.json")
    for k in range(6):
        e = j.add(entry(a=0.1 * k))
        j.rate(e["id"], 1 if k % 2 else -1)
    t = j.taste()
    assert len(t.ratings) == 6
