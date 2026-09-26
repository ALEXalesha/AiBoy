"""Гитара в окне: панель с грифом, звук попытки, x1 во время игры, 👍/👎, «Попытки»,
журнал и гитара переживают перезапуск, настройки."""
import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

import paths
from game import app as app_module


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.show_page("observe")
    yield w
    w.close()


def bore(win):
    life = win.observe.life
    life.has_guitar = True
    life.mood.world, life.mood.music, life.mood.world_time = 0.0, 0.9, 60.0


def play_until(win, kind, limit=60 * 30):
    for _ in range(limit):
        life = win.observe.life
        before = len(life.guitarist.journal.entries)
        was_playing = life.playing
        win.observe.tick()
        if kind == "sit" and life.playing and not was_playing:
            return True
        if kind == "attempt" and len(life.guitarist.journal.entries) > before:
            return True
    return False


def test_sitting_down_shows_the_neck_and_plays_the_attempt(win):
    bore(win)
    assert play_until(win, "sit")
    assert win.observe.guitar_panel.isVisible()
    for _ in range(90):
        win.observe.tick()
    attempt = win.observe.life.attempt
    assert attempt is not None and win.observe.guitar_panel.neck.attempt is attempt
    assert any(str(h).startswith("попытка") for h in win.audio.history)
    assert "Учит «" in win.observe.guitar_panel.title.text() or "Свободная" in win.observe.guitar_panel.title.text()


def test_while_playing_life_runs_at_x1_unless_the_setting_says_otherwise(win):
    bore(win)
    play_until(win, "sit")
    win.change_setting("speed", 8)
    assert win.observe.effective_speed() == 1
    win.change_setting("guitar_fast", "silent")
    assert win.observe.effective_speed() == 8
    heard = len(win.audio.history)
    for _ in range(60 * 3):
        win.observe.tick()
    assert all(not str(h).startswith("попытка") for h in win.audio.history[heard:])


def test_thumbs_up_goes_into_the_attempt(win):
    bore(win)
    play_until(win, "sit")
    for _ in range(90):
        win.observe.tick()
    QTest.mouseClick(win.observe.guitar_panel.like, Qt.MouseButton.LeftButton)
    assert "войдёт" in win.observe.guitar_panel.result.text()
    assert play_until(win, "attempt")
    entry = win.observe.life.guitarist.journal.entries[-1]
    assert entry["rating"] == 1
    assert win.stats.likes == 1 and win.stats.attempts == 1


def test_attempts_page_lists_listens_and_compares(win):
    bore(win)
    play_until(win, "sit")
    for _ in range(3):
        assert play_until(win, "attempt")
    win.show_page("attempts")
    page = win.attempts_page
    assert page.table.rowCount() == len(win.observe.life.guitarist.journal.entries)
    QTest.mouseClick(page.listen_button, Qt.MouseButton.LeftButton)
    newest = win.observe.life.guitarist.journal.entries[-1]
    assert win.audio.history[-1] == f"попытка {newest['id']}"
    QTest.mouseClick(page.compare_button, Qt.MouseButton.LeftButton)
    first = win.observe.life.guitarist.journal.first_of(newest.get("tune"))
    assert win.audio.history[-1] == f"сравнение {first['id']} и {newest['id']}"
    QTest.mouseClick(page.dislike, Qt.MouseButton.LeftButton)
    assert win.observe.life.guitarist.journal.get(newest["id"])["rating"] == -1
    assert page.tune.count() >= 1


def test_guitar_journal_and_musician_survive_a_restart(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show_page("observe")
    bore(w)
    play_until(w, "sit")
    assert play_until(w, "attempt")
    seed = w.observe.life.body_seed
    attempts = w.observe.life.guitarist.musician.attempts
    w.close()
    again = app_module.MainWindow(autostart=False)
    try:
        life = again.observe.life
        assert life.body_seed == seed and life.has_guitar
        assert len(life.guitarist.journal.entries) == 1
        assert life.guitarist.musician.attempts == attempts
    finally:
        again.close()


def test_guitar_settings_are_saved(win):
    for name, value in (("guitar_mode", "hands"), ("guitar_freq", "often"), ("volume", 0)):
        win.change_setting(name, value)
    saved = json.loads(open(paths.user_file("settings.json"), encoding="utf-8").read())
    assert (saved["guitar_mode"], saved["guitar_freq"], saved["volume"]) == ("hands", "often", 0)
    assert win.observe.life.guitarist.mode == "hands" and win.observe.life.mood.freq == "often"


def changed_pixels(a, b, step=2):
    n = 0
    for x in range(0, a.width(), step):
        for y in range(0, a.height(), step):
            if a.pixel(x, y) != b.pixel(x, y):
                n += 1
    return n


def test_guitar_is_drawn_lying_on_his_back_and_in_his_hands(win):
    life = win.observe.life
    view = win.observe.view
    win.observe.advance(20)
    far = life.guitar_x
    life.guitar_x = life.human.px + life.world.width / 2      # за краем экрана
    bare = view.grab().toImage()
    life.guitar_x = life.human.px + 0.8
    lying = view.grab().toImage()
    assert changed_pixels(bare, lying) > 20
    life.guitar_x = far
    life.has_guitar = True
    back = view.grab().toImage()
    assert changed_pixels(bare, back) > 20
    bore(win)
    assert play_until(win, "sit")
    for _ in range(30):
        win.observe.tick()
    playing = view.grab().toImage()
    life.has_guitar = False
    life.guitar_x = life.human.px + life.world.width / 2
    no_guitar = view.grab().toImage()
    assert changed_pixels(no_guitar, playing) > 20


def test_stats_show_the_tunes_table_only_after_he_played(win):
    win.show_page("stats")
    assert win.stats_page.tunes.isHidden()
    win.show_page("observe")
    bore(win)
    play_until(win, "sit")
    assert play_until(win, "attempt")
    win.show_page("stats")
    page = win.stats_page
    assert win.stats.best                      # первая попытка - всегда мелодия, не свободная игра
    assert not page.tunes.isHidden() and page.tunes.rowCount() == len(win.stats.best)
    assert page.big["attempts"].text() == "1"
