"""Стартовый экран: окно открывается на меню, кнопки ведут на свои экраны и обратно,
пока открыто меню - жизнь стоит."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from game import app as app_module
from game import theme


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    yield w
    w.close()


def distinct_colors(image, step=9):
    return len({image.pixel(x, y) for x in range(0, image.width(), step) for y in range(0, image.height(), step)})


def click(button):
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def test_window_opens_on_the_menu_without_the_tabs(win):
    assert win.current_page() == "menu"
    assert not win.nav.isVisible()
    assert win.menu.title.text() == "AiBoy"


def test_first_launch_offers_play_and_later_continue_and_new_world(win):
    m = win.menu
    assert m.play_button.text() == "Играть" and not m.new_button.isVisible()
    click(m.play_button)
    assert win.current_page() == "observe" and win.nav.isVisible()
    win.observe.advance(30)
    click(win.menu_button)
    assert win.current_page() == "menu"
    assert m.play_button.text() == "Продолжить" and m.new_button.isVisible()


def test_saved_state_means_continue_from_the_start(qapp):
    w = app_module.MainWindow(autostart=False)
    w.observe.advance(20)
    w.close()
    again = app_module.MainWindow(autostart=False)
    again.show()
    try:
        assert again.menu.play_button.text() == "Продолжить" and again.menu.new_button.isVisible()
    finally:
        again.close()


@pytest.mark.parametrize("button, page", [("play_button", "observe"), ("gallery_button", "gallery"),
                                          ("stats_button", "stats"), ("settings_button", "settings")])
def test_every_menu_button_leads_to_its_screen_and_back(win, button, page):
    click(getattr(win.menu, button))
    assert win.current_page() == page
    click(win.menu_button)
    assert win.current_page() == "menu"
    click(getattr(win.menu, button))
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert win.current_page() == "menu"


def test_new_world_from_the_menu(win):
    click(win.menu.play_button)
    click(win.menu_button)
    before = (win.observe.life.world_seed, len(win.gallery.entries))
    click(win.menu.new_button)
    assert win.current_page() == "observe"
    assert win.observe.life.world_seed != before[0] and len(win.gallery.entries) == before[1] + 1


def test_exit_closes_the_window(win):
    click(win.menu.quit_button)
    assert not win.isVisible()


def test_escape_on_the_menu_stays_on_the_menu(win):
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert win.current_page() == "menu" and win.isVisible()


def test_life_stands_still_while_the_menu_is_open(qapp):
    w = app_module.MainWindow(autostart=True)
    w.show()
    try:
        life = w.observe.life
        QTest.qWait(250)
        for _ in range(20):
            w.observe.on_frame(1 / 60)
        assert life.time == 0.0 and not w.observe.timer.isActive()
        w.show_page("observe")
        QTest.qWait(250)
        assert life.time > 0.0 and w.observe.timer.isActive()
        w.show_page("menu")
        t = life.time
        QTest.qWait(200)
        assert life.time == t
    finally:
        w.close()


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_menu_shows_the_current_world_behind_and_fades_to_the_theme(win, theme_name):
    win.change_setting("theme", theme_name)
    win.show_page("menu")
    QTest.qWait(10)
    image = win.grab().toImage()
    assert distinct_colors(image) > 40
    corner = image.pixelColor(2, image.height() - 2)
    assert corner.name().lstrip("#") == theme.palette(theme_name)["bg"].lstrip("#")


def test_new_worlds_do_not_repeat_recent_names(win):
    for _ in range(app_module.RECENT_NAMES + 3):
        win.observe.new_world()
    recent = win.gallery.entries[:app_module.RECENT_NAMES]
    assert len({e["world"] for e in recent}) == len(recent)
    win.observe.new_human()
    humans = [e["human"] for e in win.gallery.entries[:app_module.RECENT_NAMES]]
    assert humans[0] not in humans[1:]


ABOUT_WORDS = ("нейросет", "команд", "любопытств", "чат", "галере")


@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_menu_tells_what_this_is_and_it_is_fully_visible_at_minimum_size(qapp, scale):
    from PySide6.QtWidgets import QLabel
    w = app_module.MainWindow(autostart=False)
    w.show()
    try:
        about = w.menu.about
        text = about.text().lower()
        assert all(word in text for word in ABOUT_WORDS), text
        assert 3 <= len([ln for ln in about.text().split("\n") if ln.strip()]) <= 5
        assert about.wordWrap() and about.maximumHeight() == 16777215
        for lb in w.menu.findChildren(QLabel):
            px = lb.font().pixelSize() if lb.font().pixelSize() > 0 else round(lb.font().pointSizeF() * 96 / 72)
            lb.setStyleSheet(f"font-size: {round(px * scale)}px;")
        w.resize(w.minimumSize())
        QTest.qWait(30)
        assert about.heightForWidth(about.width()) <= about.height()
        top_left = about.mapTo(w.menu, about.rect().topLeft())
        assert top_left.y() >= 0 and top_left.y() + about.height() <= w.menu.height()
        for b in (w.menu.play_button, w.menu.quit_button):
            y = b.mapTo(w.menu, b.rect().bottomLeft()).y()
            assert y <= w.menu.height(), b.text()
    finally:
        w.close()
