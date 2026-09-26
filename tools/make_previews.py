"""Кадры экранов для README: docs/preview_*.png.

    python tools\\make_previews.py

Окно создаётся offscreen (Qt рисует в память), кадр снимается widget.grab() - без экрана
и без чужих окон поверх. Данные - во временной папке, мозг - стартовый из models/, так что
на кадрах то, что увидит человек при первом запуске: несколько миров, фразы в чате,
серия кадров наблюдения подряд.
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


def main():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(autostart=False)
    window.change_setting("volume", 0)
    window.change_setting("phrase_freq", "often")
    window.resize(*SIZE)
    window.show()
    obs = window.observe

    worlds_grid(window)

    # главный мир: пожил минуту, в чате фразы
    obs.start(MAIN_WORLD, MAIN_BODY)
    obs.advance(60 * 60)
    for theme in ("dark", "light"):
        window.change_setting("theme", theme)
        window.show_page("observe")
        obs.advance(60 * 2)
        shot(window, "observe" + ("" if theme == "dark" else "_light"))
    window.change_setting("theme", "dark")

    # серия кадров наблюдения: каждые полсекунды, с мыслями
    frames, caps = [], []
    for i in range(6):
        obs.advance(30)
        frames.append(obs.view.grab().toImage().scaled(600, 360, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                       Qt.TransformationMode.SmoothTransformation))
        caps.append(f"{obs.life.time:.1f} с")
    strip(frames, 3, "walk", caps)

    window.show_page("gallery")
    shot(window, "gallery")
    # статистика длиннее окна - кадр с высоким окном, чтобы влезла целиком
    window.resize(SIZE[0], 1060)
    window.show_page("stats")
    shot(window, "stats")
    window.resize(*SIZE)
    window.show_page("settings")
    shot(window, "settings")
    window.close()


if __name__ == "__main__":
    main()
