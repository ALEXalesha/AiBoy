"""Окно AiBoy: наблюдение, галерея, статистика, настройки. Интерфейс на русском.

Человек здесь только смотрит: мир, человечка и его фразы придумывают сети, окно их
рисует, играет звуки, сохраняет галерею, статистику и веса мозга.
"""
import os
import time

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import paths
from body.physics import DT
from game import models, theme
from game.chart import CurveChart, Sparkline, time_label
from game.chat import ChatView, clock
from game.gallery import Gallery, make_entry
from game.icon import app_icon
from game.life import Life
from game.settings import FREQ_NAMES, SIZE_NAMES, SPEEDS, WORLD_WIDTH, Settings
from game.sound import Player
from game.stats import Stats
from game.version import VERSION
from game.view import WorldView, thumbnail
from game.window_state import WindowMemory
from world.interest import score as interest_score

TITLE = "AiBoy"
WINDOW_OPTS = {"width": 1240, "height": 780, "minWidth": 900, "minHeight": 600}
FRAME_MS = 16
MAX_TICKS_PER_FRAME = 12
SAVE_BRAIN_EVERY = 60.0         # с реального времени
CURVE_EVERY = 10.0              # с прожитого - точка кривой обучения
PAGES = (("observe", "Наблюдение"), ("gallery", "Галерея"), ("stats", "Статистика"), ("settings", "Настройки"))


def label(text, name=None, wrap=False):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def card():
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def button(text, name=None, slot=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


def segmented(options, slot, name="seg"):
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    group = QButtonGroup(box)
    group.setExclusive(True)
    buttons = {}
    for value, text in options.items():
        b = button(text, name)
        b.setCheckable(True)
        b.clicked.connect(lambda _=False, v=value: slot(v))
        group.addButton(b)
        row.addWidget(b)
        buttons[value] = b
    return box, buttons


def meters(m):
    return f"{m / 1000:.2f} км".replace(".", ",") if m >= 1000 else f"{m:.0f} м"


def plural(n, one, few, many):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


class Page(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def on_show(self):
        pass

    def apply_theme(self, name):
        pass


# --- наблюдение ---


class ObservePage(Page):
    def __init__(self, window):
        super().__init__(window)
        s = window.settings
        self.life = None
        self.entry_id = None
        self.entry_base = (0.0, 0.0, 0)
        self.speed = s.speed
        self.paused = False
        self.acc = 0.0
        self.counted_time = 0.0
        self.counted_dist = 0.0
        self.curve_clock = 0.0
        self.rng = np.random.default_rng()

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(10)
        chat_card = card()
        chat_card.setFixedHeight(150)
        cl = QVBoxLayout(chat_card)
        cl.setContentsMargins(2, 2, 2, 2)
        self.chat = ChatView()
        cl.addWidget(self.chat)
        left.addWidget(chat_card)
        self.view = WorldView(window.colors())
        self.view.show_thoughts = s.show_thoughts
        left.addWidget(self.view, 1)
        root.addLayout(left, 1)

        side = card()
        side.setFixedWidth(280)
        col = QVBoxLayout(side)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(6)
        col.addWidget(label("Мир", "muted"))
        self.world_name = label("", "worldName")
        col.addWidget(self.world_name)
        self.human_name = label("", "muted")
        col.addWidget(self.human_name)
        col.addSpacing(6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        self.values = {}
        for r, (key, text) in enumerate((("time", "Живёт"), ("distance", "Пройдено"), ("phrases", "Фраз"),
                                         ("sounds", "Звуков"), ("jumps", "Прыжков"), ("falls", "Падений"))):
            grid.addWidget(label(text, "muted"), r, 0)
            self.values[key] = label("0", "value")
            grid.addWidget(self.values[key], r, 1, Qt.AlignmentFlag.AlignRight)
        col.addLayout(grid)
        col.addSpacing(8)
        col.addWidget(label("Любопытство", "section"))
        col.addWidget(label("ошибка его предсказателя: чем выше, тем больше нового", "hint", wrap=True))
        self.spark = Sparkline()
        col.addWidget(self.spark)
        col.addSpacing(8)
        self.pause_button = button("Пауза", None, self.toggle_pause)
        col.addWidget(self.pause_button)
        col.addWidget(label("Скорость", "muted"))
        box, self.speed_buttons = segmented({k: f"x{k}" for k in SPEEDS}, self.set_speed)
        col.addWidget(box)
        col.addSpacing(6)
        col.addWidget(button("Новый мир", "primary", lambda: self.new_world()))
        col.addWidget(button("Новый человечек", None, lambda: self.new_human()))
        col.addStretch(1)
        self.learn_note = label("", "hint", wrap=True)
        col.addWidget(self.learn_note)
        root.addWidget(side)
        self.speed_buttons[self.speed].setChecked(True)

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self.on_frame)
        self.clock = QElapsedTimer()
        self.panel_clock = 0.0

    # --- миры ---
    def start(self, world_seed, body_seed, width=None, entry_id=None):
        w = self.window_
        self.flush_life()
        width = width or WORLD_WIDTH[w.settings.world_size]
        self.life = Life(w.models, world_seed, body_seed, width, w.brain, frequency=w.settings.phrase_freq)
        self.counted_time = self.counted_dist = 0.0
        self.view.set_life(self.life)
        self.chat.clear_messages(f"{self.life.body.name} в мире {self.life.world.name}. Здесь появятся его фразы")
        if entry_id is None:
            interest = interest_score(self.life.world)["total"]
            e = make_entry(world_seed, body_seed, width, self.life.world.name, self.life.body.name, interest)
            w.gallery.add(e)
            w.save_gallery()
            w.stats.add_world()
            w.save_stats()
            entry_id = e["id"]
        self.entry_id = entry_id
        e = w.gallery.get(entry_id)
        self.entry_base = (e.get("lived", 0.0), e.get("distance", 0.0), int(e.get("phrases", 0))) if e else (0, 0, 0)
        self.refresh_panel()

    def new_world(self, seed=None):
        body = self.life.body_seed if self.life else int(self.rng.integers(1, 2 ** 31 - 1))
        self.start(int(seed if seed is not None else self.rng.integers(1, 2 ** 31 - 1)), body)

    def new_human(self, seed=None):
        world = self.life.world_seed if self.life else int(self.rng.integers(1, 2 ** 31 - 1))
        width = self.life.world.width if self.life else None
        self.start(world, int(seed if seed is not None else self.rng.integers(1, 2 ** 31 - 1)), width)

    def flush_life(self):
        """Итоги текущей жизни - в статистику и галерею."""
        life, w = self.life, self.window_
        if life is None:
            return
        w.stats.add_life(life.time - self.counted_time, life.distance - self.counted_dist)
        self.counted_time, self.counted_dist = life.time, life.distance
        self.update_entry()
        w.save_stats()
        w.save_gallery()

    def update_entry(self):
        """Итоги мира в галерее: то, что было до открытия, плюс эта жизнь."""
        life, w = self.life, self.window_
        if life is None or not self.entry_id or w.gallery.get(self.entry_id) is None:
            return
        lived, dist, phr = self.entry_base
        w.gallery.update(self.entry_id, lived=round(lived + life.time, 1), distance=round(dist + life.distance, 1),
                         phrases=phr + len(life.phrases))

    # --- время ---
    def set_running(self, on):
        if on:
            self.clock.start()
            self.timer.start()
        else:
            self.timer.stop()

    def toggle_pause(self):
        self.paused = not self.paused
        self.view.paused = self.paused
        self.pause_button.setText("Продолжить" if self.paused else "Пауза")
        self.view.update()

    def set_speed(self, k):
        self.speed = k
        self.speed_buttons[k].setChecked(True)

    def on_frame(self, dt=None):
        """Кадр таймера: прожить столько шагов физики, сколько прошло времени * скорость."""
        if dt is None:
            dt = min(0.1, self.clock.restart() / 1000.0) if self.clock.isValid() else FRAME_MS / 1000
        self.view.advance(dt if not self.paused else 0.0)
        if not self.paused:
            self.acc += dt * self.speed
            n = 0
            while self.acc >= DT and n < MAX_TICKS_PER_FRAME:
                self.acc -= DT
                n += 1
                self.tick()
            if n == MAX_TICKS_PER_FRAME:
                self.acc = 0.0
        self.panel_clock += dt
        if self.panel_clock >= 0.5:
            self.panel_clock = 0.0
            self.refresh_panel()
        self.window_.maybe_save_brain()
        self.view.update()

    def advance(self, ticks):
        """Прожить ticks шагов сразу, без таймера - для проверок, самопроверки и кадров."""
        for _ in range(ticks):
            if not self.paused:
                self.tick()
            self.view.advance(DT)
        self.refresh_panel()
        self.view.update()

    def tick(self):
        life, w = self.life, self.window_
        if life is None:
            return
        life.frequency = w.settings.phrase_freq
        for kind, *data in life.tick():
            if kind == "phrase":
                self.chat.add(life.time, data[0])
                w.stats.add_phrase(data[0])
            elif kind == "sound":
                w.player.play(data[0])
                w.stats.add_sound()
            elif kind in ("jump", "fall"):
                w.stats.add_event(kind)
        self.curve_clock += DT
        if self.curve_clock >= CURVE_EVERY:
            self.curve_clock = 0.0
            value = life.take_reward()
            w.stats.add_life(life.time - self.counted_time, life.distance - self.counted_dist)
            self.counted_time, self.counted_dist = life.time, life.distance
            if value is not None:
                w.stats.add_curve(w.stats.lived, value)
            self.update_entry()

    def refresh_panel(self):
        life = self.life
        if life is None:
            return
        self.world_name.setText(life.world.name)
        self.human_name.setText(f"человечек {life.body.name}")
        v = self.values
        v["time"].setText(clock(life.time))
        v["distance"].setText(meters(life.distance))
        v["phrases"].setText(str(len(life.phrases)))
        v["sounds"].setText(str(life.sounds))
        v["jumps"].setText(str(life.jumps))
        v["falls"].setText(str(life.falls))
        self.spark.set_values(life.curiosity_log[-120:])
        b = life.brain
        self.learn_note.setText(f"Мозг учится на ходу: {b.steps:,} шагов обучения".replace(",", " "))

    def apply_theme(self, name):
        self.view.colors = theme.palette(name)
        self.spark.set_theme(name)


# --- галерея ---


class GalleryPage(Page):
    THUMB = (264, 150)

    def __init__(self, window):
        super().__init__(window)
        self.cache = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        top = QHBoxLayout()
        top.addWidget(label("Галерея", "heading"))
        top.addStretch(1)
        self.count_label = label("", "muted")
        top.addWidget(self.count_label)
        outer.addLayout(top)
        outer.addWidget(label("Каждый новый мир с человечком попадает сюда сам. Открыть - прожить в нём "
                              "снова; удалить - только из галереи.", "hint", wrap=True))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid_host.setObjectName("galleryGrid")
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(14)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.scroll.setWidget(self.grid_host)
        outer.addWidget(self.scroll, 1)
        self.cards = {}
        self.confirm = self.ask

    def ask(self, text):
        box = QMessageBox(self)
        box.setWindowTitle("Удалить из галереи")
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def on_show(self):
        self.refresh()

    def pixmap(self, e):
        key = (e["id"], self.window_.settings.theme)
        if key not in self.cache:
            m = self.window_.models
            from body.cppn import generate_body
            from world.generate import generate
            world = generate(m.world_genome, e["world_seed"], e["width"])
            body = generate_body(m.body_genome, e["body_seed"])
            img = thumbnail(world, body, *self.THUMB, self.window_.colors(), t=1.0)
            self.cache[key] = QPixmap.fromImage(img)
        return self.cache[key]

    def refresh(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cards = {}
        entries = self.window_.gallery.entries
        n = len(entries)
        self.count_label.setText(f"{n} {plural(n, 'мир', 'мира', 'миров')}")
        cols = max(1, (self.width() - 60) // (self.THUMB[0] + 30))
        current = self.window_.observe.entry_id
        for i, e in enumerate(entries):
            c = QFrame()
            c.setObjectName("thumb")
            lay = QVBoxLayout(c)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(4)
            pic = QLabel()
            pic.setPixmap(self.pixmap(e))
            lay.addWidget(pic)
            title = f"{e['world']} · {e['human']}" + ("  (сейчас)" if e["id"] == current else "")
            lay.addWidget(label(title, "section"))
            n_ph = int(e.get("phrases", 0))
            lay.addWidget(label(f"прожил {clock(e.get('lived', 0))} · {n_ph} {plural(n_ph, 'фраза', 'фразы', 'фраз')}"
                                f" · интересность {e.get('interest', 0):.2f}", "hint"))
            row = QHBoxLayout()
            row.addWidget(button("Открыть", "primary", lambda _=False, i_=e["id"]: self.open_entry(i_)))
            row.addWidget(button("Удалить", "danger", lambda _=False, i_=e["id"]: self.delete_entry(i_)))
            lay.addLayout(row)
            self.grid.addWidget(c, i // cols, i % cols)
            self.cards[e["id"]] = c

    def open_entry(self, entry_id):
        e = self.window_.gallery.get(entry_id)
        if e is None:
            return False
        self.window_.observe.start(e["world_seed"], e["body_seed"], e["width"], entry_id=entry_id)
        self.window_.show_page("observe")
        return True

    def delete_entry(self, entry_id, confirm=None):
        e = self.window_.gallery.get(entry_id)
        if e is None:
            return False
        ask = confirm or self.confirm
        if not ask(f"Удалить мир «{e['world']}» с человечком {e['human']} из галереи?"):
            return False
        self.window_.gallery.remove(entry_id)
        if self.window_.observe.entry_id == entry_id:
            self.window_.observe.entry_id = None
        self.window_.save_gallery()
        self.refresh()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self.refresh()

    def apply_theme(self, name):
        if self.isVisible():
            self.refresh()


# --- статистика ---


def table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    t.setShowGrid(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return t


def fit_height(t):
    h = t.horizontalHeader().height() + 4
    for r in range(t.rowCount()):
        h += t.rowHeight(r)
    t.setFixedHeight(h)


class StatsPage(Page):
    def __init__(self, window):
        super().__init__(window)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("page")
        scroll.setWidget(inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)
        col = QVBoxLayout(inner)
        col.setContentsMargins(24, 16, 24, 16)
        col.setSpacing(14)
        col.addWidget(label("Статистика", "heading"))
        row = QHBoxLayout()
        row.setSpacing(12)
        self.big = {}
        for key, text in (("worlds", "миров"), ("lived", "прожито"), ("distance", "пройдено"),
                          ("phrases", "фраз"), ("sounds", "звуков"), ("jumps", "прыжков"), ("falls", "падений")):
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(14, 10, 14, 10)
            self.big[key] = label("0", "bigNumber")
            cl.addWidget(self.big[key])
            cl.addWidget(label(text, "muted"))
            row.addWidget(c)
        col.addLayout(row)

        learn = card()
        ll = QVBoxLayout(learn)
        ll.setContentsMargins(18, 14, 18, 14)
        ll.addWidget(label("Как учится мозг", "section"))
        ll.addWidget(label("Средняя награда любопытства по прожитому времени: ошибка предсказателя, "
                           "делённая на её среднее, плюс немного за то, что стоит на ногах.", "hint", wrap=True))
        self.chart = CurveChart()
        ll.addWidget(self.chart)
        self.brain_note = label("", "hint", wrap=True)
        ll.addWidget(self.brain_note)
        col.addWidget(learn)

        two = QHBoxLayout()
        two.setSpacing(14)
        words = card()
        wl = QVBoxLayout(words)
        wl.setContentsMargins(18, 14, 18, 14)
        wl.addWidget(label("Самые частые слова", "section"))
        self.words = table(["Слово", "Сколько раз"])
        wl.addWidget(self.words)
        wl.addStretch(1)
        two.addWidget(words, 1)
        best = card()
        bl = QVBoxLayout(best)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.addWidget(label("Самые интересные миры", "section"))
        self.best = table(["Мир", "Человечек", "Интересность", "Прожил"])
        bl.addWidget(self.best)
        bl.addWidget(label("Интересность - та же мера, по которой эволюция отбирала сеть мира: "
                           "перепады высот, проходимость, контраст, сущности.", "hint", wrap=True))
        bl.addStretch(1)
        two.addWidget(best, 2)
        col.addLayout(two)
        col.addStretch(1)

    def on_show(self):
        self.refresh()

    def refresh(self):
        w = self.window_
        w.observe.flush_life()
        s = w.stats
        self.big["worlds"].setText(str(s.worlds))
        self.big["lived"].setText(clock(s.lived))
        self.big["distance"].setText(meters(s.distance))
        for k in ("phrases", "sounds", "jumps", "falls"):
            self.big[k].setText(str(getattr(s, k)))
        self.chart.set_data(s.curve)
        b = w.brain
        pp = models.passport("brain").get("honest_training_time", {})
        pre = (f" Стартовые веса учились при сборке {pp.get('wall_minutes', '?')} мин "
               f"({pp.get('lived_hours', '?')} ч жизни в {pp.get('worlds', '?')} мирах).") if pp else ""
        self.brain_note.setText(f"Всего шагов обучения мозга: {b.steps:,}.".replace(",", " ") + pre)
        top = s.top_words(10)
        self.words.setRowCount(len(top))
        for r, (word, n) in enumerate(top):
            self.words.setItem(r, 0, QTableWidgetItem(word))
            self.words.setItem(r, 1, QTableWidgetItem(str(n)))
        fit_height(self.words)
        best = w.gallery.most_interesting(5)
        self.best.setRowCount(len(best))
        for r, e in enumerate(best):
            for c, text in enumerate((e["world"], e["human"], f"{e.get('interest', 0):.2f}", clock(e.get("lived", 0)))):
                self.best.setItem(r, c, QTableWidgetItem(text))
        fit_height(self.best)

    def apply_theme(self, name):
        self.chart.set_theme(name)


# --- настройки ---


class SettingsPage(Page):
    def __init__(self, window):
        super().__init__(window)
        s = window.settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        outer.setSpacing(14)
        outer.addWidget(label("Настройки", "heading"))
        box = card()
        box.setMaximumWidth(760)
        grid = QGridLayout(box)
        grid.setContentsMargins(24, 20, 24, 20)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(18)
        grid.setColumnStretch(1, 1)

        def row(r, title, hint, widget, extra=None):
            names = QVBoxLayout()
            names.setSpacing(2)
            names.addWidget(label(title, "section"))
            names.addWidget(label(hint, "hint", wrap=True))
            grid.addLayout(names, r, 0)
            line = QHBoxLayout()
            line.addWidget(widget, 1)
            if extra is not None:
                line.addWidget(extra)
            grid.addLayout(line, r, 1)

        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(s.volume)
        self.volume_label = label(f"{s.volume}%")
        self.volume_label.setMinimumWidth(48)
        self.volume.valueChanged.connect(self.on_volume)
        row(0, "Громкость", "Его голос: короткие мягкие звуки", self.volume, self.volume_label)

        self.freq = self.combo(FREQ_NAMES, s.phrase_freq, "phrase_freq")
        row(1, "Частота фраз", "Как часто он может говорить и издавать звуки", self.freq)
        self.speed = self.combo({k: f"x{k}" for k in SPEEDS}, s.speed, "speed")
        row(2, "Скорость по умолчанию", "С какой скоростью идёт жизнь при запуске", self.speed)
        self.thoughts = QCheckBox("Показывать")
        self.thoughts.setChecked(s.show_thoughts)
        self.thoughts.toggled.connect(lambda on: window.change_setting("show_thoughts", on))
        row(3, "Мысли", "Подписи желаний мозга над головой: куда идти, прыгнуть ли, сказать ли", self.thoughts)
        self.theme = self.combo(theme.THEME_NAMES, s.theme, "theme")
        row(4, "Тема", "Тёмная или светлая рамка вокруг мира", self.theme)
        self.size = self.combo(SIZE_NAMES, s.world_size, "world_size")
        row(5, "Размер мира", "Ширина новых миров; мир - кольцо, края нет", self.size)
        outer.addWidget(box)
        outer.addWidget(label("Всё сохраняется сразу.", "hint"))
        outer.addStretch(1)
        outer.addWidget(label(f"AiBoy {VERSION} · все сети свои, на numpy", "hint"))

    def combo(self, options, value, setting):
        c = QComboBox()
        c.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for key, text in options.items():
            c.addItem(text, key)
        c.setCurrentIndex(list(options).index(value))
        c.currentIndexChanged.connect(lambda i: self.window_.change_setting(setting, c.itemData(i)))
        return c

    def on_volume(self, v):
        self.volume_label.setText(f"{v}%")
        self.window_.change_setting("volume", v)


# --- окно ---


class MainWindow(QMainWindow):
    def __init__(self, autostart=True):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.setWindowIcon(app_icon())
        self.settings_path = paths.user_file("settings.json")
        self.stats_path = paths.user_file("stats.json")
        self.gallery_path = paths.user_file("gallery.json")
        self.brain_path = paths.user_file("brain.npz")
        self.settings = Settings.load(self.settings_path)
        self.stats = Stats.load(self.stats_path)
        self.gallery = Gallery.load(self.gallery_path)
        self.models = models.load()
        self.brain, self.brain_from_user = models.user_brain(self.brain_path)
        self.player = Player(os.path.join(paths.data_dir(), "sounds"), self.settings.volume)
        self.last_brain_save = time.monotonic()

        root = QWidget()
        root.setObjectName("root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(0)
        nav = QHBoxLayout()
        nav.setContentsMargins(16, 0, 16, 0)
        brand = label("AiBoy", "worldName")
        nav.addWidget(brand)
        nav.addSpacing(18)
        box, self.tabs = segmented(dict(PAGES), self.show_page, name="tab")
        nav.addWidget(box)
        nav.addStretch(1)
        lay.addLayout(nav)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.observe = ObservePage(self)
        self.gallery_page = GalleryPage(self)
        self.stats_page = StatsPage(self)
        self.settings_page = SettingsPage(self)
        self.pages = {"observe": self.observe, "gallery": self.gallery_page, "stats": self.stats_page,
                      "settings": self.settings_page}
        for page in self.pages.values():
            self.stack.addWidget(page)
        self.apply_theme(self.settings.theme)
        self.setMinimumSize(WINDOW_OPTS["minWidth"], WINDOW_OPTS["minHeight"])
        self.memory = WindowMemory(self, paths.user_file("window.json"), WINDOW_OPTS)
        # последний мир из галереи - или новый
        last = self.gallery.entries[0] if self.gallery.entries else None
        if last:
            self.observe.start(last["world_seed"], last["body_seed"], last["width"], entry_id=last["id"])
        else:
            self.observe.new_world()
        self.show_page("observe")
        if autostart:
            self.observe.set_running(True)

    def colors(self):
        return theme.palette(self.settings.theme)

    def current_page(self):
        current = self.stack.currentWidget()
        return next(name for name, page in self.pages.items() if page is current)

    def show_page(self, name):
        self.stack.setCurrentWidget(self.pages[name])
        self.tabs[name].setChecked(True)
        self.pages[name].on_show()

    def change_setting(self, name, value):
        setattr(self.settings, name, value)
        self.settings.save(self.settings_path)
        if name == "theme":
            self.apply_theme(value)
            self.gallery_page.cache.clear()
        elif name == "volume":
            self.player.set_volume(value)
        elif name == "show_thoughts":
            self.observe.view.show_thoughts = value
            self.observe.view.update()
        elif name == "speed":
            self.observe.set_speed(value)

    def apply_theme(self, name):
        self.setStyleSheet(theme.qss(name))
        for page in self.pages.values():
            page.apply_theme(name)
        self.observe.view.update()

    def save_stats(self):
        return self.stats.save(self.stats_path)

    def save_gallery(self):
        return self.gallery.save(self.gallery_path)

    def save_brain(self):
        try:
            self.brain.save(self.brain_path)
            self.last_brain_save = time.monotonic()
            return True
        except OSError:
            return False

    def maybe_save_brain(self):
        if time.monotonic() - self.last_brain_save > SAVE_BRAIN_EVERY:
            self.save_brain()
            self.observe.flush_life()

    # место окна
    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "memory"):
            self.memory.track()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "memory"):
            self.memory.track()

    def closeEvent(self, event):
        self.observe.set_running(False)
        self.observe.flush_life()
        self.save_brain()
        self.memory.save()
        self.settings.save(self.settings_path)
        super().closeEvent(event)


def run(argv=None):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(argv or [])
    app.setApplicationName("AiBoy")
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.memory.show()
    return app.exec()
