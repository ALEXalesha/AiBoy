import json

import pytest
from PySide6.QtCore import Qt
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
    win.show_page("observe")
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
    saved = json.loads(open(paths.user_file("settings.json"), encoding="utf-8").read())
    assert saved["speed"] == 2 and saved["show_thoughts"] is False


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


# Урок крестиков-ноликов: переносимую надпись (wordWrap) нельзя ограничивать по высоте.
# На настоящем экране текст чуть шире, чем в offscreen, фраза переносится на вторую строку
# и обрезается. Шрифт крупнее на 10% изображает этот «чуть шире».
QWIDGETSIZE_MAX = 16777215


def wrapped_labels(widget):
    from PySide6.QtWidgets import QLabel
    return [lb for lb in widget.findChildren(QLabel) if lb.wordWrap() and lb.isVisible() and lb.text()]


def test_wrapping_labels_are_never_capped_in_height(win):
    win.change_setting("phrase_freq", "often")
    win.observe.advance(60 * 12)
    for name in ["menu"] + [n for n, _ in app_module.PAGES]:
        win.show_page(name)
        QTest.qWait(10)
        capped = [lb.text() for lb in wrapped_labels(win) if lb.maximumHeight() < QWIDGETSIZE_MAX]
        assert capped == [], (name, capped)


@pytest.mark.parametrize("page", ["menu"] + [name for name, _ in app_module.PAGES])
@pytest.mark.parametrize("scale", [1.0, 1.1])
def test_text_fits_at_minimum_window_size(qapp, page, scale):
    from PySide6.QtWidgets import QLabel
    w = app_module.MainWindow(autostart=False)
    w.show()
    w.change_setting("phrase_freq", "often")
    w.observe.advance(60 * 12)
    w.show_page(page)
    qapp.processEvents()
    for lb in w.findChildren(QLabel):
        px = lb.font().pixelSize() if lb.font().pixelSize() > 0 else round(lb.font().pointSizeF() * 96 / 72)
        lb.setStyleSheet(f"font-size: {round(px * scale)}px;")
    w.resize(w.minimumSize())
    QTest.qWait(30)
    clipped = [(lb.text(), lb.height(), lb.heightForWidth(lb.width())) for lb in wrapped_labels(w.pages[page])
               if lb.heightForWidth(lb.width()) > lb.height()]
    w.close()
    assert clipped == []


def card_gaps(card):
    """Вертикальные зазоры между соседними блоками карточки и внутренние поля, px."""
    lay = card.layout()
    rects = [lay.itemAt(i).geometry() for i in range(lay.count())]
    gaps = [b.top() - a.bottom() - 1 for a, b in zip(rects, rects[1:])]
    pads = [rects[0].top(), card.height() - 1 - rects[-1].bottom(),
            min(r.left() for r in rects), card.width() - 1 - max(r.right() for r in rects)]
    return gaps, pads


@pytest.mark.parametrize("minimal", [False, True])
def test_gallery_cards_have_room_to_breathe(qapp, minimal):
    """Жалоба «слишком близко элементы»: между картинкой, названием, строкой итогов и
    кнопками - не меньше 8 px, от краёв карточки - не меньше 12 px, и в самом маленьком окне."""
    w = app_module.MainWindow(autostart=False)
    w.show()
    try:
        for seed in (1, 2, 3):
            w.observe.new_world(seed=seed)
        if minimal:
            w.resize(w.minimumSize())
        w.show_page("gallery")
        QTest.qWait(30)
        assert len(w.gallery_page.cards) == 4
        for card in w.gallery_page.cards.values():
            gaps, pads = card_gaps(card)
            assert min(gaps) >= 8, gaps
            assert min(pads) >= 12, pads
    finally:
        w.close()


def steps_for(obs, speed, frames=60):
    obs.set_speed(speed)
    start = obs.life.brain.steps
    for _ in range(frames):
        obs.on_frame(1 / 60)
    return obs.life.brain.steps - start


def test_speed_x8_lives_eight_times_faster_in_batches_without_lag(win):
    import time
    win.show_page("observe")
    obs = win.observe
    assert 8 in obs.speed_buttons
    one = steps_for(obs, 1)
    obs.set_speed(8)
    started = time.perf_counter()
    eight = steps_for(obs, 8)
    per_frame = (time.perf_counter() - started) / 60
    assert 7.5 * one <= eight <= 8.5 * one
    assert per_frame < 0.03                       # кадр считается быстрее, чем нужен экрану


def test_slow_frames_do_not_lose_time_at_x8(win):
    """Кадр пришёл через 50 мс (окно занято) - шаги не теряются, а досчитываются пачкой."""
    win.show_page("observe")
    obs = win.observe
    obs.set_speed(8)
    start = obs.life.time
    for _ in range(20):
        obs.on_frame(0.05)
    for _ in range(10):
        obs.on_frame(0.0)
    assert obs.life.time - start == pytest.approx(20 * 0.05 * 8, abs=0.1)


def test_chosen_speed_is_remembered(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show_page("observe")
    QTest.mouseClick(w.observe.speed_buttons[8], Qt.MouseButton.LeftButton)
    assert w.settings.speed == 8
    w.close()
    again = app_module.MainWindow(autostart=False)
    try:
        assert again.observe.speed == 8 and again.observe.speed_buttons[8].isChecked()
    finally:
        again.close()


def test_gallery_pictures_are_never_squeezed(qapp):
    w = app_module.MainWindow(autostart=False)
    w.show()
    try:
        for seed in range(1, 9):
            w.observe.new_world(seed=seed)
        w.resize(w.minimumSize())
        w.show_page("gallery")
        QTest.qWait(30)
        from PySide6.QtWidgets import QLabel
        for card in w.gallery_page.cards.values():
            pic = next(lb for lb in card.findChildren(QLabel) if lb.pixmap() and not lb.pixmap().isNull())
            assert (pic.width(), pic.height()) == app_module.GalleryPage.THUMB
    finally:
        w.close()
