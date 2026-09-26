import json

import pytest
from PySide6.QtTest import QTest

import paths
from game import app as app_module
from game import theme


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    yield w
    w.close()


def distinct_colors(image, step=9):
    seen = set()
    for x in range(0, image.width(), step):
        for y in range(0, image.height(), step):
            seen.add(image.pixel(x, y))
    return len(seen)


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_every_page_draws_in_both_themes(win, theme_name):
    win.change_setting("theme", theme_name)
    win.observe.advance(30)
    bg = theme.palette(theme_name)["bg"].lstrip("#")
    for name, _ in app_module.PAGES:
        win.show_page(name)
        QTest.qWait(10)
        image = win.grab().toImage()
        assert not image.isNull()
        assert distinct_colors(image) > 12, name
        corner = image.pixelColor(2, image.height() - 2)
        assert corner.name().lstrip("#") == bg, (name, corner.name())


def test_world_view_shows_a_world_not_an_empty_box(win):
    win.observe.advance(10)
    image = win.observe.view.grab().toImage()
    assert distinct_colors(image, step=5) > 60


def test_start_puts_a_new_world_into_the_gallery_and_stats(win):
    assert len(win.gallery.entries) == 1 and win.stats.worlds == 1
    win.observe.new_world(seed=123)
    assert len(win.gallery.entries) == 2 and win.stats.worlds == 2
    assert win.gallery.entries[0]["world_seed"] == 123
    saved = json.loads(open(paths.user_file("gallery.json"), encoding="utf-8").read())
    assert saved[0]["world_seed"] == 123


def test_chat_fills_with_his_phrases(win):
    win.change_setting("phrase_freq", "often")
    win.observe.advance(60 * 30)
    life = win.observe.life
    assert win.observe.chat.count() == len(life.phrases) >= 2
    assert win.observe.chat.texts()[-1].lower() == life.phrases[-1][1].lower()
    assert win.stats.phrases == len(life.phrases)


def test_pause_stops_life_and_speed_multiplies_it(win):
    obs = win.observe
    obs.toggle_pause()
    t = obs.life.time
    obs.advance(100)
    for _ in range(10):
        obs.on_frame(0.016)
    assert obs.life.time == t
    obs.toggle_pause()
    obs.set_speed(1)
    steps = obs.life.brain.steps
    for _ in range(30):
        obs.on_frame(1 / 60)
    slow = obs.life.brain.steps - steps
    obs.set_speed(4)
    steps = obs.life.brain.steps
    for _ in range(30):
        obs.on_frame(1 / 60)
    fast = obs.life.brain.steps - steps
    assert 12 <= slow <= 18 and 55 <= fast <= 65


def test_new_human_keeps_the_world(win):
    obs = win.observe
    world, body = obs.life.world_seed, obs.life.body_seed
    obs.new_human(seed=body + 1)
    assert obs.life.world_seed == world and obs.life.body_seed == body + 1


def test_gallery_opens_an_old_world_again(win):
    obs = win.observe
    obs.new_world(seed=11)
    obs.new_world(seed=22)
    win.show_page("gallery")
    old = next(e for e in win.gallery.entries if e["world_seed"] == 11)
    assert old["id"] in win.gallery_page.cards
    assert win.gallery_page.open_entry(old["id"])
    assert win.current_page() == "observe"
    assert obs.life.world_seed == 11 and obs.entry_id == old["id"]


def test_gallery_deletes_only_after_confirmation(win):
    win.observe.new_world(seed=33)
    entry = win.gallery.entries[0]
    win.show_page("gallery")
    assert not win.gallery_page.delete_entry(entry["id"], confirm=lambda text: False)
    assert win.gallery.get(entry["id"])
    asked = []
    assert win.gallery_page.delete_entry(entry["id"], confirm=lambda text: asked.append(text) or True)
    assert "удалить" in asked[0].lower()
    assert win.gallery.get(entry["id"]) is None
    saved = json.loads(open(paths.user_file("gallery.json"), encoding="utf-8").read())
    assert entry["id"] not in [e["id"] for e in saved]


def test_gallery_remembers_how_long_he_lived(win):
    obs = win.observe
    obs.advance(60 * 25)
    obs.flush_life()
    e = win.gallery.get(obs.entry_id)
    assert e["lived"] == pytest.approx(25.0, abs=0.2)
    assert win.stats.lived == pytest.approx(25.0, abs=0.2)


def test_stats_page_shows_numbers(win):
    win.observe.advance(60 * 12)
    win.show_page("stats")
    page = win.stats_page
    assert page.big["worlds"].text() == "1"
    assert page.best.rowCount() == 1


def test_settings_are_applied_and_saved(win):
    win.change_setting("show_thoughts", False)
    assert win.observe.view.show_thoughts is False
    win.change_setting("speed", 2)
    assert win.observe.speed == 2
    win.change_setting("volume", 15)
    assert win.player.volume == 15
    saved = json.loads(open(paths.user_file("settings.json"), encoding="utf-8").read())
    assert saved["volume"] == 15 and saved["speed"] == 2


def test_broken_files_do_not_stop_the_window(qapp, _own_data_dir):
    _own_data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("settings.json", "stats.json", "gallery.json", "window.json"):
        (_own_data_dir / name).write_text("{обрыв", encoding="utf-8")
    (_own_data_dir / "brain.npz").write_bytes(b"not a zip")
    w = app_module.MainWindow(autostart=False)
    try:
        assert w.settings.theme == "dark" and w.stats.worlds == 1
        assert w.brain_from_user is False
        w.observe.advance(20)
    finally:
        w.close()


def test_brain_survives_closing_the_window(qapp):
    w = app_module.MainWindow(autostart=False)
    w.observe.advance(200)
    steps = w.brain.steps
    w.close()
    again = app_module.MainWindow(autostart=False)
    try:
        assert again.brain_from_user and again.brain.steps == steps
        assert again.observe.life.world_seed == w.observe.life.world_seed
    finally:
        again.close()
