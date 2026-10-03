"""Кадры экранов для README: docs/preview_*.png.

    python tools\\make_previews.py

Окно создаётся offscreen (Qt рисует в память), кадр снимается widget.grab() - без экрана
и без чужих окон поверх. Данные - во временной папке, мозг - стартовый из models/, так что
на кадрах то, что увидит человек при первом запуске: меню, несколько миров, фразы в чате,
«что видит», серия кадров наблюдения подряд; гитара - он играет (гриф с пальцами и
крупно человечек с гитарой), «Попытки» после 300 попыток, «Научить мелодии» с записанной
мелодией, настройки с громкостью и режимом гитары.

2.0: игрушки в деле (серия), собака, птицы, гора (лезет по стене, флаг на вершине, гора с флагом
издали), нырок в маске, «Каталог» и рисунки в «Галерее».
"""
import os
import sys
import tempfile

os.environ["AIBOY_HOME"] = tempfile.mkdtemp(prefix="aiboy-previews-")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

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
    c = life.choice
    c.acts["explore"].liking, c.acts["guitar"].liking, c.since = 0.0, 0.9, 60.0
    life.decide_clock = 5.0
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


def view_frame(obs, w=600, h=360):
    return obs.view.grab().toImage().scaled(w, h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                            Qt.TransformationMode.SmoothTransformation)


def force(life, name):
    c = life.choice
    for n in c.acts:
        if n not in ("eat", "drink", "rest"):
            c.acts[n].liking = 0.0
    c.acts[name].liking, c.acts[name].boredom = 1.0, 0.0


def toys_strip(window):
    """Игрушки в деле: кубики, пузыри, самолётик, бумеранг, велосипед, фломастер."""
    obs = window.observe
    frames, caps = [], []
    titles = {"blocks": "кубики", "bubbles": "пузыри", "plane": "самолётик", "boomerang": "бумеранг",
              "bicycle": "велосипед", "marker": "фломастер"}
    ready = {"blocks": lambda t: len(t.attempt.placed) >= 4, "bubbles": lambda t: t.attempt.bubble is not None
             and t.attempt.bubble.age > 0.6, "plane": lambda t: t.attempt.t > 0.8,
             "boomerang": lambda t: t.attempt.t > 1.2, "bicycle": lambda t: t.attempt.t > 3,
             "marker": lambda t: t.attempt.t > 3.5}
    for name in titles:
        obs.start(MAIN_WORLD, MAIN_BODY)
        life = obs.life
        for _ in range(60 * 300):
            force(life, name)
            obs.advance(1)
            t = life.toy
            if t is not None and getattr(t, "attempt", None) is not None and ready[name](t):
                break
        frames.append(view_frame(obs))
        caps.append(titles[name])
    strip(frames, 3, "toys", caps)


def animals_shots(window):
    """Собака рядом с ним и стайка птиц."""
    obs = window.observe
    obs.start(5, 7)
    life = obs.life
    dog = life.spawn_dog(near=True)
    dog.trust, dog.tamed = 1.0, True     # ручная и хочет быть с ним (своя воля - обычно ещё и гулять)
    dog.choice.acts["follow"].liking, dog.choice.current = 1.0, "follow"
    for _ in range(90):                  # полторы секунды: новичок-собака бегает медленнее, чем он ходит
        dog.choice.acts["follow"].liking = 1.0
        obs.advance(1)
    shot(obs.view, "dog")
    obs.start(6, 7)                      # птицы - в своём мире, без собаки
    life = obs.life
    f = life.sky.spawn(life.human.px + 20.0, 0.8)
    f.facing = -1
    for b in f.birds:
        b.facing, b.x = -1, life.human.px + 3 + (b.x - f.x) * 0.5
    for _ in range(60 * 20):
        obs.advance(1)
        seen = [b for b in f.birds if abs(b.x - life.human.px) < 6
                and 1.0 < b.y - life.world.height_at(b.x) < 3.5]       # в кадре, а не над его краем
        if len(seen) >= 2:
            break
    shot(obs.view, "birds")


def mountain_shots(window):
    """Гора: лезет по стене с ледорубом, флаг и радость на вершине, гора с флагом издали."""
    obs = window.observe
    obs.start(9106, 7, 1280.0)
    life = obs.life
    p = life.world.mountains[0]

    def put(x):
        h = life.human
        h.px, h.py, h.vy = x, life.world.height_at(x) + h.body.leg() + 0.05, 0.0
    frames, caps = [], []
    put(p.face_foot())
    life.climb.start_face(0, "axe")
    life.climb.y = 9.0
    life.climb._place()
    obs.advance(30)
    frames.append(view_frame(obs, 800, 480))
    caps.append("лезет по стене с ледорубом")
    life.climb.mode = None
    put(p.summit - p.steep * 0.5)
    life.climb.summit(0, life._pending)
    obs.advance(40)
    frames.append(view_frame(obs, 800, 480))
    caps.append("вершина: флаг и радость")
    p.snow = 0.9
    life.climb.mode = None
    put(p.summit - p.steep * (p.easy_len + 60.0))
    obs.advance(5)
    frames.append(view_frame(obs, 800, 480))
    caps.append("покорённая гора издали")
    strip(frames, 3, "mountain", caps)


def dive_shot(window):
    from acts import dive
    from game import models
    from world.generate import generate
    from world.water import Water
    m = models.load()
    seed = next(s for s in range(7100, 7400) if any(q.depth > 1.2 for q in Water(generate(m.world_genome, s, 160.0)).ponds))
    obs = window.observe
    obs.start(seed, 7, 160.0)
    life = obs.life
    pond = max(life.water.ponds, key=lambda q: q.depth)
    h = life.human
    h.px = (pond.left + pond.right) / 2
    h.py = life.world.height_at(h.px) + h.body.leg()
    life.swim.pond = pond
    life.swim.enter()
    life.swim.swimmer.history = [2.0] * 10
    life.choice.current = "dive"
    life.swim.put_on_mask(dive.MaskHeads(seed=1))
    for _ in range(60 * 30):
        obs.advance(1)
        d = life.swim.dive
        if d is not None and d[0] == "hold" and d[6] > 0.5:
            break
    shot(obs.view, "dive")


def catalog_and_drawings(window):
    obs = window.observe
    obs.start(MAIN_WORLD, MAIN_BODY)
    life = obs.life
    for name, secs in (("blocks", 90), ("marker", 120)):
        for _ in range(int(secs * 60)):
            force(life, name)
            obs.advance(1)
    window.show_page("catalog")
    shot(window, "catalog")
    window.show_page("gallery")
    window.gallery_page.set_mode("drawings")
    shot(window, "drawings")
    window.gallery_page.set_mode("worlds")


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

    window.change_setting("show_vision", False)
    toys_strip(window)
    animals_shots(window)
    mountain_shots(window)
    dive_shot(window)
    catalog_and_drawings(window)
    window.change_setting("show_vision", True)

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
