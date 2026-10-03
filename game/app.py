"""Окно AiBoy: наблюдение, галерея, статистика, настройки. Интерфейс на русском.

Человек здесь только смотрит: мир, человечка и его фразы придумывают сети, окно их
рисует, сохраняет галерею, статистику и веса мозга.
"""
import os
import threading
import time

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import paths
from body.physics import DT
from game import models, theme
from game.chart import CurveChart, NeedsView, Sparkline, time_label
from game.chat import ChatView, clock
from game.gallery import Gallery, make_entry
from game.icon import app_icon
from game.life import Life
from game.settings import (BIRD_NAMES, DOG_NAMES, FREQ_NAMES, GUITAR_FAST_NAMES, GUITAR_FREQ_NAMES, GUITAR_MODE_NAMES,
                           SIZE_NAMES, SPEEDS, WORLD_WIDTH, Settings)
from game.stats import Stats
from game.version import VERSION
from game.view import RingMap, WorldView, draw_scene, thumbnail
from game.window_state import WindowMemory
from game.ui import Page, button, card, label, meters, plural, segmented
from game.audio import SR, AudioOut
from game.keyboard import KeyboardPage
from game import guitar_store as gstore
from game import life_store
from game.guitar_panel import GuitarPanel
from game.attempts import AttemptsPage
from game.catalog import CatalogPage
from music.songbook import Songbook
from music.string import shared_bank
from body.cppn import generate_body
from world.generate import world_name
from world.interest import score as interest_score

TITLE = "AiBoy"
WINDOW_OPTS = {"width": 1240, "height": 780, "minWidth": 900, "minHeight": 600}
FRAME_MS = 16
FRAME_BUDGET = 0.012           # с на счёт жизни за кадр: остальное - рисованию
MAX_BACKLOG = 0.5              # с жизни, которые ещё можно досчитать пачками
SAVE_BRAIN_EVERY = 60.0         # с реального времени
CURVE_EVERY = 10.0              # с прожитого - точка кривой обучения
ABOUT = ("Этот мир и человечка в нём придумали нейросети.\n"
         "Своим телом он управляет сам - никто не даёт ему команд.\n"
         "Он учится из любопытства: ищет то, чего ещё не знает.\n"
         "Его мысли появляются в чате сверху,\n"
         "а миры и человечки сохраняются в галерее.")
CHAT_HEIGHT = 150
CHAT_HEIGHT_PLAYING = 104
RECENT_NAMES = 12             # столько последних миров не повторяют имя нового
# «Научить мелодии» с 2.0 открывается из «Каталога» (карточка гитары) - вкладок иначе не
# хватает на окно в 900 px
PAGES = (("observe", "Наблюдение"), ("catalog", "Каталог"), ("gallery", "Галерея"), ("attempts", "Попытки"),
         ("stats", "Статистика"), ("settings", "Настройки"))


# --- меню ---


class MenuPage(Page):
    """Стартовый экран. За ним - текущий мир с человечком (кадр стоит: жизнь на паузе),
    к низу он растворяется в цвете темы."""

    def __init__(self, window):
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 12, 24, 12)
        outer.addStretch(3)
        # колонка фиксированной ширины: переносимая строка внутри получает высоту по ширине
        # (с флагом выравнивания в раскладке Qt считал бы её в одну строку)
        box = QWidget()
        box.setFixedWidth(560)
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(theme.GAP_SMALL)
        self.title = label("AiBoy", "menuTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.title)
        self.about = label(ABOUT, "menuSub", wrap=True)
        self.about.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.about)
        col.addSpacing(theme.GAP_BIG)
        self.play_button = button("Играть", "menuPrimary", window.play)
        self.new_button = button("Новый мир", "menu", window.play_new)
        self.gallery_button = button("Галерея", "menu", lambda: window.show_page("gallery"))
        self.stats_button = button("Статистика", "menu", lambda: window.show_page("stats"))
        self.settings_button = button("Настройки", "menu", lambda: window.show_page("settings"))
        self.quit_button = button("Выход", "menu", window.close)
        for b in (self.play_button, self.new_button, self.gallery_button, self.stats_button,
                  self.settings_button, self.quit_button):
            col.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(box, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(4)
        self.foot = label("", "menuFoot", wrap=True)
        self.foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.foot)

    def on_show(self):
        w = self.window_
        life = w.observe.life
        saved = life is not None and (w.had_saved or life.time > 0)
        self.play_button.setText("Продолжить" if saved else "Играть")
        self.new_button.setVisible(saved)
        if life is not None:
            self.foot.setText(f"мир «{life.world.name}» · человечек {life.body.name} · прожито в этом мире "
                              f"{clock(life.time)} · версия {VERSION}")
        self.setFocus()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        rect = QRectF(0, 0, self.width(), self.height())
        colors = self.window_.colors()
        life = self.window_.observe.life
        if life is not None:
            h = life.human
            view = self.window_.observe.view
            draw_scene(p, rect, life.world, h, view.cam_x, view.cam_y, view.clock, colors)
        bg = QColor(colors["bg"])
        g = QLinearGradient(0, 0, 0, rect.height())
        for pos, a in ((0.0, 0.45), (0.45, 0.62), (0.8, 0.94), (0.93, 1.0), (1.0, 1.0)):
            c = QColor(bg)
            c.setAlphaF(a)
            g.setColorAt(pos, c)
        p.fillRect(rect, g)
        p.end()


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
        self.sounding = None          # попытка, чей звук сейчас в канале гитары
        self.lived_acc = self.wall_acc = 0.0
        self.real_speed = None        # замеренная скорость жизни (x), None - ещё не мерили
        self.counted_time = 0.0
        self.counted_dist = 0.0
        self.curve_clock = 0.0
        self.rng = np.random.default_rng()

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(theme.GAP_BIG)
        left = QVBoxLayout()
        left.setSpacing(theme.GAP)
        self.chat_card = chat_card = card()
        chat_card.setFixedHeight(CHAT_HEIGHT)
        cl = QVBoxLayout(chat_card)
        cl.setContentsMargins(2, 2, 2, 2)
        self.chat = ChatView()
        cl.addWidget(self.chat)
        left.addWidget(chat_card)
        self.guitar_panel = GuitarPanel(window)
        self.guitar_panel.hide()
        left.addWidget(self.guitar_panel)
        self.view = WorldView(window.colors())
        self.view.show_thoughts = s.show_thoughts
        self.view.show_vision = s.show_vision
        left.addWidget(self.view, 1)
        root.addLayout(left, 1)

        side = card()
        col = QVBoxLayout(side)
        col.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        col.setSpacing(theme.GAP_SMALL)
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
                                         ("jumps", "Прыжков"), ("falls", "Падений"))):
            grid.addWidget(label(text, "muted"), r, 0)
            self.values[key] = label("0", "value")
            grid.addWidget(self.values[key], r, 1, Qt.AlignmentFlag.AlignRight)
        col.addLayout(grid)
        col.addSpacing(8)
        col.addWidget(label("Настроение и нужды", "section"))
        self.needs_view = NeedsView()
        self.needs_view.setToolTip("Лицо - его настроение: среднее удовольствие от занятий за последние минуты")
        col.addWidget(self.needs_view)
        col.addSpacing(6)
        col.addWidget(label("Любопытство", "section"))
        col.addWidget(label("ошибка его предсказателя: чем выше, тем больше нового", "hint", wrap=True))
        self.ring = RingMap(window.colors())
        self.ring.setToolTip("Весь мир-кольцо: оранжевая черта - он, белая точка - сундук, цветная - гитара")
        col.addWidget(self.ring)
        self.spark = Sparkline()
        col.addWidget(self.spark)
        col.addSpacing(8)
        self.pause_button = button("Пауза", None, self.toggle_pause)
        col.addWidget(self.pause_button)
        self.vision_button = button("Что видит (V)", "toggle",
                                    lambda: window.change_setting("show_vision", self.vision_button.isChecked()))
        self.vision_button.setCheckable(True)
        self.vision_button.setChecked(s.show_vision)
        col.addWidget(self.vision_button)
        # пока он гуляет с гитарой за спиной; когда играет - такая же кнопка у 👍/👎
        self.put_button = button("Положи гитару", None, lambda: self.ask_put())
        self.put_button.setToolTip("Просьба, а не команда: он решает сам")
        self.put_button.hide()
        col.addWidget(self.put_button)
        col.addWidget(label("Скорость", "muted"))
        box, self.speed_buttons = segmented({k: f"x{k}" for k in SPEEDS},
                                            lambda k: window.change_setting("speed", k))
        col.addWidget(box)
        self.real_label = label("", "hint", wrap=True)
        self.real_label.hide()
        col.addWidget(self.real_label)
        col.addSpacing(6)
        col.addWidget(button("Новый мир", "primary", lambda: self.new_world()))
        col.addWidget(button("Новый человечек", None, lambda: self.new_human()))
        col.addStretch(1)
        self.learn_note = label("", "hint", wrap=True)
        col.addWidget(self.learn_note)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFixedWidth(292)
        side_scroll.setWidget(side)
        root.addWidget(side_scroll)
        self.speed_buttons[self.speed].setChecked(True)

        self.timer = QTimer(self)
        self.timer.setInterval(FRAME_MS)
        self.timer.timeout.connect(self.on_frame)
        self.clock = QElapsedTimer()
        self.panel_clock = 0.0

    # --- миры ---
    def start(self, world_seed, body_seed, width=None, entry_id=None):
        w = self.window_
        self.stop_life()
        width = width or WORLD_WIDTH[w.settings.world_size]
        g = gstore.load(body_seed, w.songbook, mode=w.settings.guitar_mode)
        needs, pack, choice, prefs, swimmer = life_store.load(body_seed, w.settings.guitar_freq)
        self.life = Life(w.models, world_seed, body_seed, width, w.brain, frequency=w.settings.phrase_freq,
                         navigator=w.navigator, guitarist=g, guitar_freq=w.settings.guitar_freq,   # гитара - в сундуке
                         choice=choice, needs=needs, backpack=pack, food_prefs=prefs, swimmer=swimmer,
                         toy_heads=life_store.load_toys(body_seed),
                         dog_mode=w.settings.dog, dog_saved=life_store.load_dog(world_seed),
                         climber=life_store.load_climb(body_seed)[0], summits=life_store.load_summits(world_seed),
                         bird_mode=w.settings.birds, birds_saved=life_store.load_birds(world_seed))
        self.life.climb.fear = life_store.load_climb(body_seed)[1]
        self.show_guitar(False)
        self.refresh_put()
        self.counted_time = self.counted_dist = 0.0
        self.acc = 0.0
        self.view.set_life(self.life)
        self.ring.set_life(self.life)
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

    def fresh_seed(self, name_of, key):
        """Зерно, чьё имя (его выдаёт сеть) не встречалось в последних RECENT_NAMES мирах
        галереи: имена из двух-трёх слогов у сети иногда повторяются, а третья «Эльма»
        подряд - скучно. Двадцать попыток, потом - какое вышло."""
        recent = {e[key] for e in self.window_.gallery.entries[:RECENT_NAMES]}
        seed = int(self.rng.integers(1, 2 ** 31 - 1))
        for _ in range(20):
            if name_of(seed) not in recent:
                break
            seed = int(self.rng.integers(1, 2 ** 31 - 1))
        return seed

    def on_show(self):
        # текущий мир удалён из галереи - войти в него нельзя: новый мир, и он попадает в галерею
        # (Алексей: удалил миры, в том числе текущий, вошёл - а он на месте)
        if self.life is not None and (self.entry_id is None or self.window_.gallery.get(self.entry_id) is None):
            self.new_world()

    def new_world(self, seed=None):
        m = self.window_.models
        body = self.life.body_seed if self.life else self.fresh_seed(
            lambda s: generate_body(m.body_genome, s).name, "human")
        if seed is None:
            seed = self.fresh_seed(lambda s: world_name(m.world_genome, s), "world")
        self.start(int(seed), body)

    def new_human(self, seed=None):
        m = self.window_.models
        world = self.life.world_seed if self.life else int(self.rng.integers(1, 2 ** 31 - 1))
        width = self.life.world.width if self.life else None
        if seed is None:
            seed = self.fresh_seed(lambda s: generate_body(m.body_genome, s).name, "human")
        self.start(world, int(seed), width)

    def stop_life(self):
        """Единственная остановка жизни - при любой смене мира или человечка и при закрытии
        окна: итоги в статистику и галерею, жизнь останавливается (идущая попытка брошена),
        звук её попытки затухает за FADE, гриф пустеет, журнал и веса гитары - на диск.
        Фоновой работы у жизни нет: кэш струн общий для всех жизней, он не останавливается."""
        life = self.life
        if life is None:
            return
        self.flush_life()
        life.stop()
        self.sync_sound()
        self.guitar_panel.clear()
        self.show_guitar(False)
        self.save_guitar()
        self.life = None

    def sound_wanted(self):
        """Попытка, которая должна звучать сейчас, или None: звук идёт за жизнью - пауза,
        меню, x2-x8 «без звука», новый мир - тишина."""
        life = self.life
        if life is None or not life.playing or life.attempt is None or self.paused:
            return None
        if self.window_.current_page() == "menu" or self.effective_speed() != 1:
            return None
        return life.attempt

    def sync_sound(self):
        """Привести звук гитары к жизни: другая попытка (или никакой) - старая затухает, нужная
        звучит с того места, где сейчас play_t."""
        w = self.window_
        want = self.sound_wanted()
        if want is self.sounding:
            return
        if self.sounding is not None:
            w.audio.stop("guitar")
            self.sounding = None
        if want is None:
            return
        life = self.life
        audio = w.bank.render(want.notes(), want.duration)
        start = int(life.play_t * SR) if life.play_t > 2 * DT else 0
        label = f"попытка {life.guitarist.journal.next_id}"
        if start:
            label += f" с {life.play_t:.1f} с"
        w.audio.play(audio[start:], label, channel="guitar", fade_in=start > 0)
        self.sounding = want

    def save_guitar(self):
        life = self.life
        if life is not None:
            life_store.save(life)
        if life is not None and life.guitarist is not None:
            gstore.save(life.guitarist)

    def show_guitar(self, on):
        self.guitar_panel.setVisible(on)
        self.chat_card.setFixedHeight(CHAT_HEIGHT_PLAYING if on else CHAT_HEIGHT)

    def effective_speed(self):
        """Пока он играет, жизнь идёт в x1, чтобы каждую попытку было слышно (настройка «x8 во
        время игры» может это отключить - тогда попытки на скорости идут без звука)."""
        life = self.life
        if life is not None and life.playing and self.window_.settings.guitar_fast == "slow":
            return 1
        return self.speed

    def ask_put(self):
        """«Положи гитару»: просьба к нему; ответ - в чат от сети речи."""
        life = self.life
        if life is None:
            return None
        res = life.ask_put()
        if res is None:
            return None
        agree, events = res
        self.handle(life, events)
        self.sync_sound()
        self.refresh_put()
        return agree

    def refresh_put(self):
        """Кнопки «Положи гитару» активны, только пока гитара у него в руках или за спиной."""
        on = self.life is not None and self.life.has_guitar
        self.guitar_panel.put.setEnabled(on)
        self.put_button.setEnabled(on)
        self.put_button.setVisible(on and not self.life.playing)

    def after_rating(self, rating):
        """Оценка законченной попытке (в идущую попадёт вместе с попыткой - add_attempt)."""
        self.save_guitar()
        self.window_.stats.guitar_rating(rating)

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
        self.sync_sound()
        self.view.update()

    def set_speed(self, k):
        self.speed = k
        self.speed_buttons[k].setChecked(True)
        self.sync_sound()

    def on_frame(self, dt=None):
        """Кадр таймера: прожить столько шагов физики, сколько прошло времени * скорость."""
        if dt is None:
            dt = min(0.1, self.clock.restart() / 1000.0) if self.clock.isValid() else FRAME_MS / 1000
        if self.window_.current_page() == "menu":
            return                      # пока открыто меню, жизнь стоит
        self.view.advance(dt if not self.paused else 0.0)
        if not self.paused:
            # жизнь считается пачками по DT; на кадр - не дольше FRAME_BUDGET, чтобы окно
            # успевало рисовать и на x8. Не успели - остаток досчитается в следующих кадрах
            # (но не больше MAX_BACKLOG: после долгой заминки время не нагоняется рывком).
            speed = self.effective_speed()
            self.acc = min(self.acc + dt * speed, MAX_BACKLOG * speed)
            started = time.perf_counter()
            ticks = 0
            while self.acc >= DT and time.perf_counter() - started < FRAME_BUDGET:
                self.acc -= DT
                self.tick()
                ticks += 1
            # настоящая скорость - сколько жизни прожито за сколько времени на часах: если кадр
            # не успевает, шаг физики не растёт (иначе ломаются касания), а скорость честно ниже
            self.lived_acc += ticks * DT
            self.wall_acc += dt
        if self.life is not None and self.life.playing:
            self.guitar_panel.neck.set_time(self.life.play_t)
        self.sync_sound()
        self.panel_clock += dt
        if self.panel_clock >= 0.5:
            self.panel_clock = 0.0
            if self.wall_acc > 0:
                self.real_speed = self.lived_acc / self.wall_acc
            self.lived_acc = self.wall_acc = 0.0
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
        self.handle(life, life.tick())
        if life.playing:
            w.stats.guitar_time += DT
        w.stats.add_activity(life.choice.current, DT)
        self.curve_clock += DT
        if self.curve_clock >= CURVE_EVERY:
            self.curve_clock = 0.0
            value = life.take_reward()
            w.stats.add_life(life.time - self.counted_time, life.distance - self.counted_dist)
            self.counted_time, self.counted_dist = life.time, life.distance
            if value is not None:
                w.stats.add_curve(w.stats.lived, value)
            self.update_entry()

    def handle(self, life, events):
        """События жизни - в чат, статистику, панель гитары и звук."""
        w = self.window_
        for kind, *data in events:
            if kind == "phrase":
                self.chat.add(life.time, data[0])
                w.stats.add_phrase(data[0])
            elif kind in ("jump", "fall"):
                w.stats.add_event(kind)
            elif kind == "pickup":
                self.save_guitar()
            elif kind == "sit":
                self.show_guitar(True)
            elif kind == "attempt_start":
                attempt = data[0]
                self.guitar_panel.set_attempt(attempt)
                # звук попытки - целиком в канал гитары; струны из кэша, это миллисекунды
                self.sync_sound()
            elif kind == "attempt":
                self.guitar_panel.set_result(data[0])
                w.stats.add_attempt(data[0], life.guitarist.journal.best(data[0].get("tune")))
                self.save_guitar()
            elif kind == "stand":
                self.show_guitar(False)
            elif kind == "put":
                self.save_guitar()
            elif kind == "sound":
                # звуки игрушек - только на x1-x2 (на x4-x16 пропускаются; гитара - своим каналом)
                if self.effective_speed() <= 2 and not self.paused:
                    from sound import sfx
                    audio = sfx.make(data[0], data[1], data[2])
                    if audio is not None:
                        w.audio.play(audio, data[0], channel="toys")
            elif kind == "toy_attempt":
                w.stats.attempts += 1
            if kind in ("summit", "flag", "toy_attempt", "dive", "dog_brought", "birds_came"):
                w.stats.add_life_event(kind, data)
            elif kind in ("toy_done", "put_toy"):
                self.save_guitar()
            if kind in ("pickup", "put", "sit", "stand"):
                self.refresh_put()

    def refresh_panel(self):
        life = self.life
        if life is None:
            return
        self.world_name.setText(life.world.name)
        self.human_name.setText(f"человечек {life.body.name}")
        want = self.effective_speed()
        slow = (self.real_speed is not None and not self.paused and want > 1 and self.real_speed < 0.9 * want)
        self.real_label.setVisible(slow)
        if slow:
            self.real_label.setText(f"x{want} (сейчас x{max(1, round(self.real_speed))}): кадр не успевает")
        v = self.values
        v["time"].setText(clock(life.time))
        v["distance"].setText(meters(life.distance))
        v["phrases"].setText(str(len(life.phrases)))
        v["jumps"].setText(str(life.jumps))
        v["falls"].setText(str(life.falls))
        self.spark.set_values(life.curiosity_log[-120:])
        self.needs_view.set_state(life.choice.mood, life.needs)
        self.ring.update()
        b = life.brain
        self.learn_note.setText(f"Мозг учится на ходу: {b.steps:,} шагов обучения".replace(",", " "))

    def apply_theme(self, name):
        self.view.colors = theme.palette(name)
        self.ring.colors = theme.palette(name)
        self.spark.set_theme(name)
        self.needs_view.set_theme(name)


# --- галерея ---


def picture(kind, e, size, colors):
    """Картинка рисунка (на листе) или улова (рыба, какой была) для «Галереи»."""
    from PySide6.QtGui import QPainterPath, QPolygonF
    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32)
    img.fill(QColor(colors["surface2"]))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    w, h = size
    if kind == "drawings":
        sheet = QRectF(w * 0.18, h * 0.08, w * 0.64, h * 0.84)
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.setBrush(QColor(250, 249, 244))
        p.drawRect(sheet)
        pts = (e.get("drawing") or {}).get("points") or []
        if len(pts) >= 2:
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
            sc = min(sheet.width(), sheet.height()) * 0.8 / span
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            path = QPainterPath(QPointF(sheet.center().x() + (xs[0] - cx) * sc, sheet.center().y() - (ys[0] - cy) * sc))
            for x, y in zip(xs[1:], ys[1:]):
                path.lineTo(QPointF(sheet.center().x() + (x - cx) * sc, sheet.center().y() - (y - cy) * sc))
            p.setPen(QPen(QColor(40, 55, 170), 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
    else:
        f = e.get("fish") or {}
        col = f.get("color") or [0.5, 0.6, 0.7]
        body = QColor(int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
        L = w * 0.55 * min(1.0, 0.4 + float(f.get("length", 0.3)) * 2.0)
        H = L * max(0.25, float(f.get("height", 0.1)) / max(float(f.get("length", 0.3)), 0.05))
        c = QPointF(w / 2, h / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(body)
        tail = float(f.get("tail", 0.5))
        p.drawPolygon(QPolygonF([QPointF(c.x() - L * 0.4, c.y()), QPointF(c.x() - L * (0.7 + 0.2 * tail), c.y() - H * 0.5),
                                 QPointF(c.x() - L * (0.7 + 0.2 * tail), c.y() + H * 0.5)]))
        p.drawEllipse(c, L / 2, H / 2)
        p.setBrush(QColor(20, 20, 20))
        p.drawEllipse(QPointF(c.x() + L * 0.3, c.y() - H * 0.1), max(1.5, H * 0.12), max(1.5, H * 0.12))
    p.end()
    return img


class GalleryPage(Page):
    THUMB = (264, 150)
    MODES = (("worlds", "Миры"), ("drawings", "Рисунки"), ("catch", "Улов"))

    def __init__(self, window):
        super().__init__(window)
        self.cache = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        top = QHBoxLayout()
        top.addWidget(label("Галерея", "heading"))
        self.mode = "worlds"
        self.mode_buttons = {}
        for key, text in self.MODES:
            b = button(text, None, lambda _=False, k=key: self.set_mode(k))
            b.setCheckable(True)
            b.setChecked(key == "worlds")
            self.mode_buttons[key] = b
            top.addWidget(b)
        top.addStretch(1)
        self.count_label = label("", "muted")
        top.addWidget(self.count_label)
        outer.addLayout(top)
        self.mode_hint = label("Каждый новый мир с человечком попадает сюда сам. Открыть - прожить в нём "
                               "снова; удалить - только из галереи.", "hint", wrap=True)
        outer.addWidget(self.mode_hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid_host.setObjectName("galleryGrid")
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(theme.GAP_BIG)
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

    HINTS = {"worlds": "Каждый новый мир с человечком попадает сюда сам. Открыть - прожить в нём снова; "
                       "удалить - только из галереи.",
             "drawings": "Всё, что нарисовал фломастером текущий человечек: красота - на его вкус (его сеть вкуса "
                         "плюс ваши 👍/👎 в «Попытках»).",
             "catch": "Рыбы, которых поймал на удочку текущий человечек: вид рыбы - из сети рыб того пруда."}

    def set_mode(self, mode):
        self.mode = mode
        for k, b in self.mode_buttons.items():
            b.setChecked(k == mode)
        self.mode_hint.setText(self.HINTS[mode])
        self.refresh()

    def journal_items(self):
        life = self.window_.observe.life
        if life is None:
            return []
        from toys import base as toy_base
        name = "marker" if self.mode == "drawings" else "rod"
        j = life.toy_journals.get(name) or toy_base.journal(life.body_seed, name)
        if self.mode == "drawings":
            return [e for e in j.entries if isinstance(e.get("drawing"), dict)]
        return [e for e in j.entries if e.get("caught") and e.get("fish")]

    def refresh_items(self):
        """Рисунки или улов - карточки с картинкой, датой и подписью."""
        import time as _time
        items = list(reversed(self.journal_items()))
        n = len(items)
        word = (plural(n, "рисунок", "рисунка", "рисунков") if self.mode == "drawings"
                else plural(n, "рыба", "рыбы", "рыб"))
        self.count_label.setText(f"{n} {word}")
        cols = max(1, (self.width() - 60) // (self.THUMB[0] + 2 * theme.PAD + theme.GAP_BIG + 4))
        if not items:
            self.grid.addWidget(label("Пока пусто: он рисует и рыбачит, когда сам захочет.", "hint"), 0, 0)
        for i, e in enumerate(items[:60]):
            c = QFrame()
            c.setObjectName("thumb")
            lay = QVBoxLayout(c)
            lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
            lay.setSpacing(theme.GAP)
            pic = QLabel()
            pic.setPixmap(QPixmap.fromImage(picture(self.mode, e, self.THUMB, self.window_.colors())))
            pic.setFixedSize(*self.THUMB)
            lay.addWidget(pic)
            when = _time.strftime("%d.%m %H:%M", _time.localtime(e.get("time", 0)))
            lay.addWidget(label(f"№{e.get('id', '')} · {when}", "section"))
            if self.mode == "drawings":
                lay.addWidget(label(f"красота {e.get('beauty', 0):.2f}", "hint"))
            else:
                lay.addWidget(label(f"длина {100 * float(e['fish'].get('length', 0)):.0f} см", "hint"))
            self.grid.addWidget(c, i // cols, i % cols)
            c.show()                             # сразу: Qt показывает новых детей отложенно, а скрытых сетка не считает

    def refresh(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)              # сразу, не потом: иначе старые карточки видны под новыми
                old.deleteLater()
        self.cards = {}
        if self.mode != "worlds":
            self.refresh_items()
        else:
            self.refresh_worlds()
        self.grid.activate()
        # высота сетки - под новые карточки сразу: область прокрутки подгоняет её только потом, и
        # первый кадр сжимал карточки - подпись и кнопки ложились на картинку (карточки уже показаны)
        view = self.scroll.viewport()
        self.grid_host.resize(view.width(), max(view.height(), self.grid_host.minimumSizeHint().height()))

    def refresh_worlds(self):
        entries = self.window_.gallery.entries
        n = len(entries)
        self.count_label.setText(f"{n} {plural(n, 'мир', 'мира', 'миров')}")
        cols = max(1, (self.width() - 60) // (self.THUMB[0] + 2 * theme.PAD + theme.GAP_BIG + 4))
        current = self.window_.observe.entry_id
        for i, e in enumerate(entries):
            c = QFrame()
            c.setObjectName("thumb")
            lay = QVBoxLayout(c)
            lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
            lay.setSpacing(theme.GAP)
            pic = QLabel()
            pic.setPixmap(self.pixmap(e))
            pic.setFixedSize(*self.THUMB)      # картинка, не текст: сжимать её нельзя
            lay.addWidget(pic)
            title = f"{e['world']} · {e['human']}" + ("  (сейчас)" if e["id"] == current else "")
            lay.addWidget(label(title, "section"))
            n_ph = int(e.get("phrases", 0))
            lay.addWidget(label(f"прожил {clock(e.get('lived', 0))} · {n_ph} {plural(n_ph, 'фраза', 'фразы', 'фраз')}"
                                f" · интересность {e.get('interest', 0):.2f}", "hint"))
            row = QHBoxLayout()
            row.setSpacing(theme.GAP)
            row.addWidget(button("Открыть", "primary", lambda _=False, i_=e["id"]: self.open_entry(i_)))
            row.addWidget(button("Удалить", "danger", lambda _=False, i_=e["id"]: self.delete_entry(i_)))
            lay.addLayout(row)
            self.grid.addWidget(c, i // cols, i % cols)
            c.show()                             # сразу: Qt показывает новых детей отложенно, а скрытых сетка не считает
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
        col.setSpacing(theme.GAP_BIG)
        col.addWidget(label("Статистика", "heading"))
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        self.big = {}
        for key, text in (("worlds", "миров"), ("lived", "прожито"), ("distance", "пройдено"),
                          ("phrases", "фраз"), ("jumps", "прыжков"), ("attempts", "попыток")):
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
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

        gcard = card()
        gl = QVBoxLayout(gcard)
        gl.setContentsMargins(18, 14, 18, 14)
        gl.addWidget(label("Гитара", "section"))
        self.guitar_line = label("", "muted", wrap=True)
        gl.addWidget(self.guitar_line)
        self.tunes = table(["Мелодия", "Лучшая похожесть"])
        gl.addWidget(self.tunes)
        self.taste_line = label("", "hint", wrap=True)
        gl.addWidget(self.taste_line)
        col.addWidget(gcard)

        acard = card()
        al = QVBoxLayout(acard)
        al.setContentsMargins(18, 14, 18, 14)
        al.addWidget(label("Занятия", "section"))
        self.fav_line = label("", "muted", wrap=True)
        al.addWidget(self.fav_line)
        self.acts = table(["Занятие", "Время", "Доля"])
        al.addWidget(self.acts)
        self.acts_lines = label("", "hint", wrap=True)
        al.addWidget(self.acts_lines)
        col.addWidget(acard)

        two = QHBoxLayout()
        two.setSpacing(theme.GAP_BIG)
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
        for k in ("phrases", "jumps", "attempts"):
            self.big[k].setText(str(getattr(s, k)))
        self.guitar_line.setText(f"С гитарой - {clock(s.guitar_time)}, попыток {s.guitar_attempts}, "
                                 f"оценок 👍 {s.likes} и 👎 {s.dislikes}, падений в мире - {s.falls}.")
        best = sorted(s.best.items(), key=lambda kv: -kv[1])
        self.tunes.setRowCount(len(best))
        for r, (tune, v) in enumerate(best):
            self.tunes.setItem(r, 0, QTableWidgetItem(tune))
            self.tunes.setItem(r, 1, QTableWidgetItem(f"{v:.2f}"))
        fit_height(self.tunes)
        self.tunes.setVisible(bool(best))           # пока не играл - без пустой шапки таблицы
        life = w.observe.life
        acc = None
        if life is not None and life.guitarist is not None:
            acc = life.guitarist.taste.accuracy()
        self.taste_line.setText("Вкус угадывает ваши оценки: " + (f"{acc * 100:.0f} % (догадка - до того, как "
                                "увидел оценку)" if acc is not None else "пока оценок мало") + ".")
        self.refresh_activities(s, life)
        self.chart.set_data(s.curve)
        b = w.brain
        pp = models.passport("brain").get("honest_training_time", {})
        pre = (f" Стартовые веса учились при сборке {pp.get('wall_minutes', '?')} мин: эволюционная стратегия "
               f"({pp.get('es_generations', '?')} поколений) и онлайн.") if pp else ""
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

    ACT_NAMES = {"explore": "Гулять", "guitar": "Гитара", "swim": "Плавать", "eat": "Есть", "drink": "Пить",
                 "rest": "Отдыхать", "dog": "Собака", "climb": "Гора", "dive": "Нырять в маске"}

    def refresh_activities(self, s, life):
        """Сколько времени на что ушло, любимое занятие и итоги занятий 2.0."""
        from toys import base as toy_base
        from toys import registry
        names = dict(self.ACT_NAMES)
        names.update({t.name: t.title for t in registry.TOYS.values()})
        total = sum(s.activity.values()) or 1.0
        rows = sorted(s.activity.items(), key=lambda kv: -kv[1])[:12]
        self.acts.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            for c, text in enumerate((names.get(k, k), clock(v), f"{100 * v / total:.0f} %")):
                self.acts.setItem(r, c, QTableWidgetItem(text))
        fit_height(self.acts)
        self.acts.setVisible(bool(rows))
        fav = s.favourite_activity()
        self.fav_line.setText(f"Любимое занятие (дольше всего): {names.get(fav, fav)}." if fav else
                              "Пока он больше гуляет - любимое занятие ещё не определилось.")
        parts = [f"поймал рыб - {s.fish}", f"нарисовал - {s.drawings}", f"вершин - {s.summits}, флагов - {s.flags}",
                 f"нырков - {s.dives}, рыб под водой видел - {s.fish_seen}", f"птиц прилетало - {s.birds_seen}",
                 f"собака приносила мяч - {s.dog_brought}"]
        if life is not None:
            j = life.toy_journals.get("blocks") or toy_base.journal(life.body_seed, "blocks")
            best = max([int(e.get("blocks", 0)) for e in j.entries] or [0])
            parts.append(f"самая высокая башня - {best} кубиков")
            sw = life.swim.swimmer
            parts.append("плавает: " + ("уже пловец" if sw.can_swim() else f"учится ({len(sw.history)} попыток)"))
            if life.dog is not None:
                parts.append(f"доверие собаки - {life.dog.trust:.2f}" + (" (ручная)" if life.dog.tamed else ""))
            parts.append(f"полёт птиц этого мира: {life.sky.flyer.flights} пробных полётов")
        self.acts_lines.setText("; ".join(parts) + ".")

    def apply_theme(self, name):
        self.chart.set_theme(name)


# --- настройки ---


class SettingsPage(Page):
    def __init__(self, window):
        super().__init__(window)
        s = window.settings
        # в прокрутке: в низком окне строки не сжимаются, а уезжают вниз
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("page")
        scroll.setWidget(inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)
        outer = QVBoxLayout(inner)
        outer.setContentsMargins(24, 16, 24, 16)
        outer.setSpacing(theme.GAP_BIG)
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

        self.freq = self.combo(FREQ_NAMES, s.phrase_freq, "phrase_freq")
        row(0, "Частота фраз", "Как часто он может говорить в чат", self.freq)
        self.speed = self.combo({k: f"x{k}" for k in SPEEDS}, s.speed, "speed")
        row(1, "Скорость по умолчанию", "С какой скоростью идёт жизнь при запуске", self.speed)
        self.thoughts = QCheckBox("Показывать")
        self.thoughts.setChecked(s.show_thoughts)
        self.thoughts.toggled.connect(lambda on: window.change_setting("show_thoughts", on))
        self.vision = QCheckBox("Показывать")
        self.vision.setChecked(s.show_vision)
        self.vision.toggled.connect(lambda on: window.change_setting("show_vision", on))
        row(5, "Что видит", "Поверх мира: что он видит, где ему интересно, куда хочет идти (клавиша V)",
            self.vision)
        row(2, "Мысли", "Подписи желаний мозга над головой: куда идти, прыгнуть ли, сказать ли", self.thoughts)
        self.theme = self.combo(theme.THEME_NAMES, s.theme, "theme")
        row(3, "Тема", "Тёмная или светлая рамка вокруг мира", self.theme)
        self.size = self.combo(SIZE_NAMES, s.world_size, "world_size")
        row(4, "Размер мира", "Ширина новых миров; мир - кольцо, края нет", self.size)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(s.volume)
        self.volume_label = label(f"{s.volume}%")
        self.volume_label.setMinimumWidth(48)
        self.volume.valueChanged.connect(self.on_volume)
        row(6, "Громкость", "Гитара и другие игрушки; 0 - звуковое устройство не открывается вовсе. "
            "Голоса у человечка нет", self.volume, self.volume_label)
        self.guitar_mode = self.combo(GUITAR_MODE_NAMES, s.guitar_mode, "guitar_mode")
        row(7, "Гитара: как играет", "«Музыкант» - сеть сама выбирает струну и лад, руки только показывают. "
            "«Руками» - нота звучит, только если он дотянулся до лада и струны", self.guitar_mode)
        self.guitar_freq = self.combo(GUITAR_FREQ_NAMES, s.guitar_freq, "guitar_freq")
        row(8, "Как часто берёт гитару", "Сдвигает порог: насколько мир должен надоесть сильнее музыки",
            self.guitar_freq)
        self.guitar_fast = self.combo(GUITAR_FAST_NAMES, s.guitar_fast, "guitar_fast")
        row(9, "Скорость, пока играет", "x2-x8 и гитара вместе: слушать каждую попытку или не сбавлять скорость",
            self.guitar_fast)
        self.dog = self.combo(DOG_NAMES, s.dog, "dog")
        row(10, "Собака", "Придёт ли в мир собака: никогда, иногда (около 40 % миров) или часто. В хорошем "
            "настроении он встречает её чаще", self.dog)
        self.birds = self.combo(BIRD_NAMES, s.birds, "birds")
        row(11, "Птицы", "Как часто прилетают стайки; в хорошем настроении - чаще и больше (до 8)", self.birds)
        outer.addWidget(box)
        outer.addWidget(label("Всё сохраняется сразу.", "hint"))
        outer.addStretch(1)
        outer.addWidget(label(f"AiBoy {VERSION} · все сети свои, на numpy", "hint"))

    def on_volume(self, v):
        self.volume_label.setText(f"{v}%")
        self.window_.change_setting("volume", v)

    def combo(self, options, value, setting):
        c = QComboBox()
        c.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for key, text in options.items():
            c.addItem(text, key)
        c.setCurrentIndex(list(options).index(value))
        c.currentIndexChanged.connect(lambda i: self.window_.change_setting(setting, c.itemData(i)))
        return c



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
        self.navigator_path = paths.user_file("navigator.npz")
        self.navigator = models.user_navigator(self.navigator_path)
        self.last_brain_save = time.monotonic()
        self.songbook = Songbook.load(paths.user_file("songbook.json"))
        self.audio = AudioOut(self.settings.volume)
        self.bank = shared_bank()
        # звук струн - в фоне, пока человек смотрит меню: первая попытка уже не ждёт синтеза.
        # Без экрана (тесты, самопроверка) - по требованию: нота синтезируется за 10 мс
        if not self.bank.warming and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            self.bank.warming = True
            threading.Thread(target=self.bank.warm, daemon=True).start()

        root = QWidget()
        root.setObjectName("root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(0)
        self.nav = QWidget()
        nav = QHBoxLayout(self.nav)
        nav.setContentsMargins(16, 0, 16, 0)
        self.menu_button = button("В меню", "tab", lambda: self.show_page("menu"))
        nav.addWidget(self.menu_button)
        nav.addSpacing(6)
        brand = label("AiBoy", "worldName")
        nav.addWidget(brand)
        nav.addSpacing(18)
        box, self.tabs = segmented(dict(PAGES), self.show_page, name="tab")
        nav.addWidget(box)
        nav.addStretch(1)
        lay.addWidget(self.nav)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.autostart = autostart
        self.had_saved = bool(self.gallery.entries)
        self.observe = ObservePage(self)
        self.menu = MenuPage(self)
        self.gallery_page = GalleryPage(self)
        self.stats_page = StatsPage(self)
        self.settings_page = SettingsPage(self)
        self.teach_page = KeyboardPage(self)
        self.attempts_page = AttemptsPage(self)
        self.catalog_page = CatalogPage(self)
        self.pages = {"menu": self.menu, "observe": self.observe, "catalog": self.catalog_page,
                      "gallery": self.gallery_page, "attempts": self.attempts_page, "teach": self.teach_page,
                      "stats": self.stats_page, "settings": self.settings_page}
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
        self.show_page("menu")

    def colors(self):
        return theme.palette(self.settings.theme)

    def current_page(self):
        current = self.stack.currentWidget()
        return next(name for name, page in self.pages.items() if page is current)

    def show_page(self, name):
        """Меню - во весь экран, без вкладок, и жизнь на паузе; остальные экраны - со
        вкладками, жизнь идёт (если окно запущено с таймером)."""
        in_menu = name == "menu"
        if in_menu:
            self.observe.flush_life()
        self.observe.set_running(self.autostart and not in_menu)
        self.nav.setVisible(not in_menu)
        self.stack.setCurrentWidget(self.pages[name])
        if not in_menu and name in self.tabs:
            self.tabs[name].setChecked(True)
        elif name == "teach":
            self.tabs["catalog"].setChecked(True)             # «Научить мелодии» - из «Каталога»
        self.observe.sync_sound()
        self.pages[name].on_show()

    def play(self):
        """«Играть» / «Продолжить»: к жизни в текущем мире."""
        self.show_page("observe")

    def play_new(self):
        self.observe.new_world()
        self.show_page("observe")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.current_page() != "menu":
            self.show_page("menu")
            return
        if event.key() == Qt.Key.Key_V and self.current_page() == "observe":
            self.change_setting("show_vision", not self.settings.show_vision)
            return
        super().keyPressEvent(event)

    def change_setting(self, name, value):
        setattr(self.settings, name, value)
        self.settings.save(self.settings_path)
        if name == "theme":
            self.apply_theme(value)
            self.gallery_page.cache.clear()
        elif name == "show_vision":
            self.observe.view.show_vision = value
            self.observe.vision_button.setChecked(value)
            self.settings_page.vision.setChecked(value)
            self.observe.view.update()
        elif name == "show_thoughts":
            self.observe.view.show_thoughts = value
            self.observe.view.update()
        elif name == "birds" and self.observe.life is not None:
            self.observe.life.sky.mode = value         # и в этом мире - сразу
        elif name == "dog" and self.observe.life is not None:
            self.observe.life.dog_mode = value
        elif name == "speed":
            self.observe.set_speed(value)
        elif name == "guitar_fast":
            self.observe.sync_sound()
        elif name == "volume":
            self.audio.set_volume(value)
        elif name == "guitar_mode":
            life = self.observe.life
            if life is not None and life.guitarist is not None:
                life.guitarist.mode = value
        elif name == "guitar_freq":
            life = self.observe.life
            if life is not None:
                life.choice.freq = value

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
            self.navigator.save(self.navigator_path)
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
        self.observe.stop_life()
        self.audio.close()
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
