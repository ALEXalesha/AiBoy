"""«Научить мелодии»: клавиши пианино (мышь и клавиатура компьютера), запись, слушать,
название, сохранить - мелодия встаёт в песенник рядом со встроенными."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from game import app as app_module
from game.keyboard import KEYMAP, Recorder


def test_recorder_turns_presses_into_eighths_and_rests():
    r = Recorder(tempo=120)                    # восьмая = 0.25 с
    r.start(0.0)
    r.press(60, 0.0)
    r.release(60, 0.5)
    r.press(64, 0.75)
    r.release(64, 1.75)
    r.press(67, 1.75)
    assert r.stop(2.26) == ((60, 2), (None, 1), (64, 4), (67, 2))


def test_new_key_ends_the_previous_note_one_voice():
    r = Recorder(tempo=120)
    r.start(0.0)
    r.press(60, 0.0)
    r.press(62, 0.5)                           # первая не отпущена - кончается здесь
    r.release(60, 0.9)                         # поздно отпущенная первая ничего не меняет
    r.release(62, 1.0)
    assert r.stop(1.0) == ((60, 2), (62, 2))


def test_short_taps_are_at_least_an_eighth_and_leading_silence_is_skipped():
    r = Recorder(tempo=100)
    r.start(0.0)
    r.press(65, 1.0)
    r.release(65, 1.05)
    assert r.stop(1.2) == ((65, 1),)


def test_computer_keys_cover_two_octaves():
    assert KEYMAP[Qt.Key.Key_Z] == 60 and KEYMAP[Qt.Key.Key_M] == 71
    assert KEYMAP[Qt.Key.Key_Q] == 72 and KEYMAP[Qt.Key.Key_U] == 83
    assert sorted(KEYMAP.values()) == list(range(60, 84))


@pytest.fixture
def win(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.show_page("teach")
    yield w
    w.close()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_record_by_keyboard_listen_and_save(win, qapp):
    page = win.teach_page
    clock = Clock()
    page.clock = clock
    page.tempo.setValue(120)
    QTest.mouseClick(page.record_button, Qt.MouseButton.LeftButton)
    for key, down, up in ((Qt.Key.Key_Z, 0.0, 0.5), (Qt.Key.Key_C, 0.5, 1.0), (Qt.Key.Key_B, 1.0, 2.0)):
        clock.t = down
        QTest.keyPress(page, key)
        clock.t = up
        QTest.keyRelease(page, key)
    QTest.mouseClick(page.stop_button, Qt.MouseButton.LeftButton)
    assert page.notes == ((60, 2), (64, 2), (67, 4))
    assert "до4" in page.shown.text() and "соль4" in page.shown.text()
    QTest.mouseClick(page.listen_button, Qt.MouseButton.LeftButton)
    assert win.audio.history[-1] == "мелодия: новая"
    page.name.setText("Моя первая")
    QTest.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert win.songbook.by_name("Моя первая").notes == ((60, 2), (64, 2), (67, 4))
    win.close()
    again = app_module.MainWindow(autostart=False)
    try:
        assert again.songbook.by_name("Моя первая") is not None
    finally:
        again.close()


def test_mouse_on_the_piano_plays_and_records(win):
    page = win.teach_page
    clock = Clock()
    page.clock = clock
    QTest.mouseClick(page.record_button, Qt.MouseButton.LeftButton)
    piano = page.piano
    point = piano.key_rect(62).center().toPoint()
    QTest.mousePress(piano, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    clock.t = 0.6
    QTest.mouseRelease(piano, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseClick(page.stop_button, Qt.MouseButton.LeftButton)
    assert page.notes == ((62, 2),)
    assert win.audio.history[-1] == "клавиша 62"


def test_save_needs_a_name_and_refuses_builtin_names(win):
    page = win.teach_page
    page.notes = ((60, 2),)
    page.name.setText("")
    QTest.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert "название" in page.status.text().lower()
    page.name.setText("Чижик-пыжик")
    QTest.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert "встроен" in page.status.text().lower()
    assert len(win.songbook.own) == 0
