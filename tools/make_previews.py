"""Кадры экранов для README: docs/preview_*.png.

    python tools\\make_previews.py

Окно создаётся offscreen (Qt рисует в память), кадр снимается widget.grab() - без экрана
и без чужих окон поверх. Данные - во временной папке, мозг - стартовый из models/, так что
на кадрах то, что увидит человек при первом запуске: меню, несколько миров, фразы в чате,
«что видит», серия кадров наблюдения подряд; гитара - он играет (гриф с пальцами и
крупно человечек с гитарой), «Попытки» после 300 попыток, «Научить мелодии» с записанной
мелодией, настройки с громкостью и режимом гитары.
"""
import os
import sys
import tempfile

os.environ["AIBOY_HOME"] = tempfile.mkdtemp(prefix="aiboy-previews-")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "4")

import offscreen  # noqa: E402

offscreen.setup(force=True)

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from game.app import MainWindow  # noqa: E402
from game.view import draw_scene  # noqa: E402

SIZE = (1240, 780)
DOCS = os.path.join(ROOT, "docs")
MAIN_WORLD = int(os.environ.get("AIBOY_PREVIEW_WORLD", "7"))
MAIN_BODY = int(os.environ.get("AIBOY_PREVIEW_BODY", "12"))
OTHER = [(101, 5), (202, 8), (303, 21), (404, 3), (505, 17), (606, 40)]


def shot(widget, name):
    QApplication.processEvents()
    path = os.path.join(DOCS, f"preview_{name}.png")
    widget.grab().save(path)
    print(path)
    return path


def strip(frames, cols, name, captions):
    w, h = frames[0].width(), frames[0].height()
    rows = (len(frames) + cols - 1) // cols
    gap = 8
    out = QImage(cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap, QImage.Format.Format_ARGB32)
    out.fill(QColor("#111419"))
    p = QPainter(out)
    font = QFont(p.font())
    font.setPixelSize(15)
    font.setWeight(QFont.Weight.DemiBold)
    p.setFont(font)
    for i, (img, cap) in enumerate(zip(frames, captions)):
        x, y = (i % cols) * (w + gap), (i // cols) * (h + gap)
        p.drawImage(x, y, img)
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(QRectF(x + 11, y + 9, w, 22), Qt.AlignmentFlag.AlignLeft, cap)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(x + 10, y + 8, w, 22), Qt.AlignmentFlag.AlignLeft, cap)
    p.end()
    path = os.path.join(DOCS, f"preview_{name}.png")
    out.save(path)
    print(path)


def worlds_grid(window):
    """Шесть миров разных зёрен - как одна сеть придумывает непохожие миры."""
    frames, caps = [], []
    obs = window.observe
    for ws, bs in OTHER:
        obs.start(ws, bs)
        obs.advance(60 * 4)
        life = obs.life
        img = QImage(560, 315, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        h = life.human
        draw_scene(p, QRectF(0, 0, 560, 315), life.world, h, h.px, h.world.height_at(h.px) + h.body.leg(),
                   3.0, window.colors())
        p.end()
        frames.append(img)
        caps.append(f"{life.world.name} · {life.body.name}")
    strip(frames, 3, "worlds", caps)


def train_guitar(window, life, attempts=300):
    """Попытки подряд без мира: как если бы он играл около 45 минут. Несколько оценок - чтобы
    на кадре были 👍/👎 и вкус."""
    g = life.guitarist
    for i in range(attempts):
        a = g.attempt(life.body)
        rating = None
        if i % 37 == 5:
            rating = 1 if i % 2 else -1
        entry, _, _ = g.finish(a, life.body, rating=rating)
        window.stats.add_attempt(entry)
        window.stats.guitar_time += a.duration


def guitar_shots(window):
    obs = window.observe
    life = obs.life
    train_guitar(window, life)
    life.has_guitar = True
    life.mood.world, life.mood.music, life.mood.world_time = 0.0, 0.9, 60.0
    window.change_setting("show_vision", False)
    window.show_page("observe")
    # садится и играет; кадр - сразу после удара по струне, чтобы горел палец на грифе
    for _ in range(60 * 60):
        obs.tick()
        if life.playing and life.attempt is not None and life.play_t > 2.0:
            obs.guitar_panel.neck.set_time(life.play_t)
            if obs.guitar_panel.neck.current()[1]:
                break
    obs.refresh_panel()
    shot(window, "guitar")
    # крупно: тот же кадр мира вдвое ближе
    img = QImage(2480, 1400, QImage.Format.Format_ARGB32)
    p = QPainter(img)
    h = life.human
    f = draw_scene(p, QRectF(0, 0, 2480, 1400), life.world, h, h.px, h.py + 0.9, 3.0, window.colors(),
                   guitar="play")
    p.end()
    cx, cy = f.X(h.px), f.Y(h.py + 0.5)
    close = img.copy(int(cx - 450), int(cy - 300), 900, 560)
    path = os.path.join(DOCS, "preview_guitar_close.png")
    close.save(path)
    print(path)
    window.change_setting("show_vision", True)
    window.show_page("attempts")
    page = window.attempts_page
    tune = next((e["tune"] for e in reversed(life.guitarist.journal.entries) if e.get("tune")), "")
    idx = page.tune.findData(tune)
    page.tune.setCurrentIndex(max(0, idx))
    shot(window, "attempts")


def teach_shot(window):
    window.show_page("teach")
    page = window.teach_page
    now = [0.0]
    page.clock = lambda: now[0]
    beat = 60.0 / page.tempo.value() / 2
    page.start_recording()
    tune = [(64, 1), (67, 1), (67, 1), (69, 1), (67, 2), (64, 2), (62, 1), (64, 1), (65, 2), (64, 4)]
    for pitch, d in tune:
        page.on_press(pitch)
        now[0] += d * beat * 0.9
        page.on_release(pitch)
        now[0] += d * beat * 0.1
    page.stop_recording()
    page.name.setText("Моя песенка")
    page.piano.down = {67}
    page.piano.update()
    shot(window, "teach")
    page.piano.down = set()


def main():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(autostart=False)
    window.change_setting("phrase_freq", "often")
    window.resize(*SIZE)
    window.show()
    obs = window.observe

    worlds_grid(window)

    # главный мир: пожил минуту, в чате фразы; сначала - меню поверх него
    obs.start(MAIN_WORLD, MAIN_BODY)
    window.show_page("observe")
    obs.advance(60 * 60)
    window.show_page("menu")
    shot(window, "menu")
    for theme in ("dark", "light"):
        window.change_setting("theme", theme)
        window.show_page("observe")
        obs.advance(60 * 2)
        shot(window, "observe" + ("" if theme == "dark" else "_light"))
    window.change_setting("theme", "dark")

    # «что видит» крупно: вид мира без панели
    shot(obs.view, "vision")
    # серия кадров наблюдения: каждые полсекунды, с мыслями, без «что видит»
    window.change_setting("show_vision", False)
    frames, caps = [], []
    for i in range(6):
        obs.advance(30)
        frames.append(obs.view.grab().toImage().scaled(600, 360, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                       Qt.TransformationMode.SmoothTransformation))
        caps.append(f"{obs.life.time:.1f} с")
    strip(frames, 3, "walk", caps)
    window.change_setting("show_vision", True)

    guitar_shots(window)
    teach_shot(window)

    window.show_page("gallery")
    shot(window, "gallery")
    # статистика длиннее окна - кадр с высоким окном, чтобы влезла целиком
    window.resize(SIZE[0], 1060)
    window.show_page("stats")
    shot(window, "stats")
    # настройки с гитарой (режим, громкость, как часто) - тоже в высоком окне
    window.show_page("settings")
    shot(window, "settings")
    window.resize(*SIZE)
    window.close()


if __name__ == "__main__":
    main()
