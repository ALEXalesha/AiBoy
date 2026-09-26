import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from game.gallery import MAX_ENTRIES, Gallery, make_entry
from game.settings import Settings
from game.stats import Stats

JSONISH = st.recursive(st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=5),
                       lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4),
                       max_leaves=12)


def test_settings_defaults_and_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings.load(path)
    assert (s.volume, s.phrase_freq, s.speed, s.theme, s.world_size) == (60, "normal", 1, "dark", "medium")
    s.volume, s.speed, s.theme, s.show_thoughts = 30, 4, "light", False
    assert s.save(path)
    again = Settings.load(path)
    assert (again.volume, again.speed, again.theme, again.show_thoughts) == (30, 4, "light", False)


def test_each_broken_setting_falls_back_alone(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"volume": 500, "speed": 3, "theme": "pink", "phrase_freq": "often",
                                "show_thoughts": "yes", "world_size": "large"}), encoding="utf-8")
    s = Settings.load(path)
    assert (s.volume, s.speed, s.theme, s.show_thoughts) == (60, 1, "dark", True)
    assert s.phrase_freq == "often" and s.world_size == "large"


@given(JSONISH)
def test_settings_survive_any_garbage(tmp_path_factory, data):
    path = tmp_path_factory.mktemp("s") / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    s = Settings.load(path)
    for name in ("volume", "phrase_freq", "speed", "show_thoughts", "theme", "world_size"):
        assert Settings.valid(name, getattr(s, name))


def test_truncated_files_give_defaults(tmp_path):
    for name, cls in (("settings.json", Settings), ("stats.json", Stats), ("gallery.json", Gallery)):
        path = tmp_path / name
        path.write_text('{"volume": 3', encoding="utf-8")
        cls.load(path)


def test_stats_count_life(tmp_path):
    s = Stats()
    s.add_world()
    s.add_life(30.0, 12.5)
    s.add_phrase("вижу синее дерево справа")
    s.add_phrase("синее дерево!")
    s.add_sound()
    s.add_event("jump")
    s.add_event("fall")
    assert (s.worlds, s.lived, s.distance, s.phrases, s.sounds, s.jumps, s.falls) == (1, 30.0, 12.5, 2, 1, 1, 1)
    assert set(s.top_words(2)) == {("синее", 2), ("дерево", 2)}
    assert "вижу" in s.words and all(len(w) >= 3 for w in s.words)
    path = tmp_path / "stats.json"
    s.save(path)
    again = Stats.load(path)
    assert again.to_dict() == s.to_dict()


def test_stats_curve_stays_short():
    s = Stats()
    for i in range(5000):
        s.add_curve(float(i), 0.1)
    assert len(s.curve) <= 400
    assert s.curve[0][0] < s.curve[-1][0]


@given(JSONISH)
def test_stats_survive_any_garbage(tmp_path_factory, data):
    path = tmp_path_factory.mktemp("st") / "stats.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    s = Stats.load(path)
    assert s.worlds >= 0 and s.lived >= 0 and all(isinstance(v, int) for v in s.words.values())


def entry(i, **kw):
    return make_entry(world_seed=i, body_seed=i + 1, width=160.0, world="Мир", human="Том", interest=0.5, **kw)


def test_gallery_adds_updates_and_removes(tmp_path):
    g = Gallery()
    e = entry(1)
    g.add(e)
    g.update(e["id"], lived=12.0, phrases=3)
    assert g.get(e["id"])["lived"] == 12.0
    path = tmp_path / "gallery.json"
    g.save(path)
    again = Gallery.load(path)
    assert again.get(e["id"])["phrases"] == 3
    assert again.remove(e["id"]) and not again.entries


def test_gallery_keeps_the_newest(tmp_path):
    g = Gallery()
    for i in range(MAX_ENTRIES + 5):
        g.add(entry(i))
    assert len(g.entries) == MAX_ENTRIES
    assert g.entries[0]["world_seed"] == MAX_ENTRIES + 4


def test_gallery_skips_broken_entries(tmp_path):
    path = tmp_path / "gallery.json"
    good = entry(5)
    path.write_text(json.dumps([good, {"id": "x"}, 7, {"id": "y", "world_seed": "a"}]), encoding="utf-8")
    g = Gallery.load(path)
    assert [e["id"] for e in g.entries] == [good["id"]]


@given(JSONISH)
def test_gallery_survives_any_garbage(tmp_path_factory, data):
    path = tmp_path_factory.mktemp("g") / "gallery.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    Gallery.load(path)


@pytest.mark.parametrize("bad", [b"\xff\xfe\x00", b"", b"[[[[[[["])
def test_binary_junk_does_not_break_loading(tmp_path, bad):
    for name, cls in (("settings.json", Settings), ("stats.json", Stats), ("gallery.json", Gallery)):
        path = tmp_path / name
        path.write_bytes(bad)
        cls.load(path)
