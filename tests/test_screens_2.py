"""Экраны 2.0: «Попытки» для всех занятий с повтором, «Галерея» - рисунки и улов, «Каталог» -
новые игрушки и занятия, «Настройки» - собака и птицы, «Статистика» - занятия."""
import time

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from game import app as app_module
from toys import registry


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.observe.new_world(seed=3)
    yield w
    w.close()


def settle(widgets_fn, limit=2.0):
    prev, deadline = None, time.monotonic() + limit
    while time.monotonic() < deadline:
        QTest.qWait(30)
        now = widgets_fn()
        if now == prev:
            return now
        prev = now
    return prev


def add_entries(life):
    j = life.journal_add("climb", {"route": "face", "height": 12.0, "share": 0.2, "falls": 2, "top": False,
                                   "success": 0.2})
    from toys import base as toy_base
    for name in ("ball", "marker", "rod"):
        if name not in life.toy_journals:
            life.toy_journals[name] = toy_base.journal(life.body_seed, name)
    life.toy_journals["ball"].add({"kind": "throw", "trace": [[0.0, 1.5], [2.0, 3.0], [4.0, 0.2]], "success": 0.6,
                                   "pleasure": 0.55})
    life.toy_journals["marker"].add({"beauty": 0.8, "success": 0.8, "drawing": {
        "points": [[0.0, 0.0], [0.2, 0.3], [0.4, 0.0], [0.2, -0.3], [0.0, 0.0]], "pens": [0.0] * 24}})
    life.toy_journals["rod"].add({"caught": True, "why": "поймал", "success": 1.0,
                                  "fish": {"length": 0.25, "height": 0.08, "tail": 0.5, "color": [0.8, 0.5, 0.2]}})
    return j


def test_attempts_show_every_activity_and_replay_a_trace(win):
    life = win.observe.life
    add_entries(life)
    win.show_page("attempts")
    page = win.attempts_page
    names = [page.activity.itemData(i) for i in range(page.activity.count())]
    for n in ("guitar", "ball", "blocks", "rod", "bicycle", "marker", "dog", "mask", "climb"):
        assert n in names, n
    assert "rope" not in names and "axe" not in names            # снаряжение - не занятие
    page.activity.setCurrentIndex(names.index("ball"))
    assert page.table.rowCount() == 1 and page.table.item(0, 2).text() == "бросил"
    page.table.selectRow(0)
    page.listen()
    assert page.replay.items and page.replay.isVisible()
    page.activity.setCurrentIndex(names.index("climb"))
    assert page.table.rowCount() == 1 and "по стене" in page.table.item(0, 2).text()
    page.activity.setCurrentIndex(names.index("guitar"))
    assert not page.replay.isVisible()


def test_gallery_shows_drawings_and_catch_in_cards_with_room(win):
    life = win.observe.life
    add_entries(life)
    for _ in range(8):                   # несколько рядов: сетка выше окна (кадр: подпись на белом листе)
        life.toy_journals["marker"].add(dict(life.toy_journals["marker"].entries[0]))
    win.resize(1280, 720)
    win.show_page("gallery")
    g = win.gallery_page
    for mode, word in (("drawings", "рисун"), ("catch", "рыб")):
        g.set_mode(mode)
        assert word in g.count_label.text()
        cards = [g.grid.itemAt(i).widget() for i in range(g.grid.count())]
        QApplication.processEvents()     # один проход событий - первый кадр, который увидит человек
        need = cards[0].minimumSizeHint().height()
        assert cards and all(c.height() >= need for c in cards)         # подпись не наезжает на картинку
        rects = settle(lambda: [c.geometry().getRect() for c in cards])
        assert all(r[3] >= g.THUMB[1] for r in rects)                   # не сжаты в полосы
    g.set_mode("worlds")
    assert g.cards


def test_world_cards_have_room_on_the_first_frame(win):
    for seed in range(4, 12):            # миров - на несколько рядов, сетка выше окна
        win.observe.new_world(seed=seed)
    win.resize(1240, 780)
    win.show_page("observe")             # как человек: с другой вкладки в «Галерею»
    QTest.qWait(0)
    win.show_page("gallery")
    cards = list(win.gallery_page.cards.values())
    assert len(cards) >= 6
    QApplication.processEvents()         # первый кадр (было: кнопки и подпись поверх кадра мира)
    need = cards[0].minimumSizeHint().height()
    assert all(c.height() >= need for c in cards)


def test_catalog_has_the_new_toys_tools_and_activities(win):
    win.show_page("catalog")
    page = win.catalog_page
    for t in registry.TOYS.values():
        c = page.cards[t.name]
        assert (c.offer is None) == t.tool                               # снаряжение не предлагают
    for n in ("dog", "climb", "dive"):
        assert n in page.cards
    page.refresh()


def test_dog_and_bird_settings_apply_at_once_and_survive(win, tmp_path):
    win.change_setting("birds", "often")
    win.change_setting("dog", "never")
    assert win.observe.life.sky.mode == "often" and win.observe.life.dog_mode == "never"
    from game.settings import Settings
    s = Settings.load(win.settings_path)
    assert s.birds == "often" and s.dog == "never"
    win.change_setting("world_size", "giant")
    assert Settings.load(win.settings_path).world_size == "giant"     # 1280 м сохраняется


def test_stats_show_time_per_activity_and_the_favourite(win):
    win.stats.add_activity("ball", 300.0)
    win.stats.add_activity("explore", 600.0)
    win.stats.add_life_event("summit", ())
    win.show_page("stats")
    page = win.stats_page
    assert page.acts.rowCount() >= 2 and "Мяч" in page.fav_line.text()
    assert "вершин - 1" in page.acts_lines.text()
