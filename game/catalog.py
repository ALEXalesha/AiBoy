"""Экран «Каталог»: игрушки из сундука и занятия - как они ему нравятся.

Карточка игрушки: как выглядит, «нравится» (полоса), попыток, лучший результат, любимая
отмечена, «Предложить» - положить игрушку рядом с ним: возьмёт или нет, решает он сам (тот
же выбор занятий с прибавкой), ответ - в чате. У гитары ещё «Научить мелодии» (свой экран).
Карточки занятий: гулять (дальше - плавать, собака). Нужды и рюкзак появляются здесь с
задачей 9 плана.

Всё берётся из жизни (выбор занятий, сундук, журналы) при каждом показе и раз в секунду,
пока экран открыт.
"""
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QProgressBar, QScrollArea, QVBoxLayout, QWidget

from game import theme
from game.chart import NeedsView
from game.ui import Page, button, card, label, plural
from toys import registry

ACT_TITLES = {"explore": "Гулять", "swim": "Плавать"}
ACT_NOTES = {"explore": "Ходит по миру и сам решает, куда: вперёд, назад, остановиться и осмотреться.",
             "swim": "Учится плавать с нуля: сначала барахтается стоймя, потом плывёт. Никогда не тонет - "
                     "устал или нахлебался - его выносит на берег."}
ICON = 96


def toy_icon(name, colors):
    """Картинка игрушки - тем же рисованием, что в мире."""
    img = QImage(ICON, ICON, QImage.Format.Format_ARGB32)
    img.fill(QColor(colors["surface2"]))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    class Frame:
        ppm = ICON / 0.9

        @staticmethod
        def X(x):
            return x * Frame.ppm + ICON / 2

        @staticmethod
        def Y(y):
            return ICON / 2 - y * Frame.ppm

        @staticmethod
        def P(pt):
            return QPointF(Frame.X(pt[0]), Frame.Y(pt[1]))
    if name == "guitar":
        from game.view import draw_guitar
        draw_guitar(p, Frame, (-0.18, -0.12), (0.62, 0.78), 1.0)
    elif name == "swim":
        from PySide6.QtGui import QPainterPath
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(colors["accent"]))
        wave = QPainterPath(QPointF(0, ICON * 0.55))
        for i in range(4):
            wave.quadTo(QPointF(ICON * (i + 0.5) / 4, ICON * 0.45), QPointF(ICON * (i + 1) / 4, ICON * 0.55))
        wave.lineTo(ICON, ICON)
        wave.lineTo(0, ICON)
        wave.closeSubpath()
        p.drawPath(wave)
        p.setBrush(QColor(colors["text"]))
        p.drawEllipse(QPointF(ICON * 0.35, ICON * 0.45), 7, 7)
        p.drawRoundedRect(ICON * 0.42, ICON * 0.47, ICON * 0.35, 8, 4, 4)
    elif name == "explore":
        # холмы и тропинка со следами - «гулять»
        from PySide6.QtGui import QPainterPath, QPen
        path = QPainterPath(QPointF(0, ICON * 0.72))
        path.cubicTo(ICON * 0.3, ICON * 0.5, ICON * 0.6, ICON * 0.85, ICON, ICON * 0.62)
        path.lineTo(ICON, ICON)
        path.lineTo(0, ICON)
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(colors["accent"]))
        p.drawPath(path)
        p.setBrush(QColor(colors["text"]))
        for i, (x, y) in enumerate(((0.18, 0.5), (0.34, 0.42), (0.5, 0.47), (0.66, 0.4), (0.82, 0.44))):
            p.drawEllipse(QPointF(ICON * x, ICON * (y - 0.08 * (i % 2))), 4, 6)
    p.end()
    return QPixmap.fromImage(img)


class ActivityCard(QWidget):
    def __init__(self, page, name, title, note, toy=None):
        super().__init__()
        self.page, self.name, self.toy = page, name, toy
        box = card()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(box)
        lay = QHBoxLayout(box)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP_BIG)
        self.icon = label("")
        self.icon.setFixedSize(ICON, ICON)
        lay.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(theme.GAP_SMALL)
        top = QHBoxLayout()
        self.title = label(title, "section")
        top.addWidget(self.title)
        self.favourite = label("любимое", "badge")
        top.addWidget(self.favourite)
        top.addStretch(1)
        col.addLayout(top)
        col.addWidget(label(note, "hint", wrap=True))
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        row.addWidget(label("нравится", "muted"))
        self.liking = QProgressBar()
        self.liking.setRange(0, 100)
        self.liking.setTextVisible(False)
        self.liking.setFixedHeight(10)
        row.addWidget(self.liking, 1)
        col.addLayout(row)
        self.numbers = label("", "muted", wrap=True)
        col.addWidget(self.numbers)
        buttons = QHBoxLayout()
        buttons.setSpacing(theme.GAP)
        self.offer = None
        if toy is not None:
            self.offer = button("Предложить", "primary", lambda: page.offer(name))
            self.offer.setToolTip("Положить рядом с ним. Возьмёт или нет - решит сам")
            buttons.addWidget(self.offer)
        if name == "guitar":
            buttons.addWidget(button("Научить мелодии", None, lambda: page.window_.show_page("teach")))
        buttons.addStretch(1)
        col.addLayout(buttons)
        lay.addLayout(col, 1)

    def refresh(self, life, favourite):
        a = life.choice.acts.get(self.name) if life is not None else None
        like = a.liking if a is not None else 0.0
        self.liking.setValue(int(round(100 * like)))
        self.favourite.setVisible(favourite == self.name)
        if self.name == "guitar" and life is not None and life.guitarist is not None:
            j = life.guitarist.journal
            n = len(j.entries)
            best = max([e["A"] for e in j.entries if e.get("A") is not None] or [None], key=lambda v: v or 0)
            where = {"chest": "в сундуке", "carried": "у него"}.get(life.inv.where.get("guitar"), "лежит в мире")
            text = f"{n} {plural(n, 'попытка', 'попытки', 'попыток')}"
            if best is not None:
                text += f", лучшая похожесть на мелодию {best:.2f}"
            self.numbers.setText(f"{text}; сейчас {where}.")
            if self.offer is not None:
                self.offer.setEnabled(life.inv.where.get("guitar") != "carried" and not life.playing)
        elif self.name == "swim" and life is not None:
            sw = life.swim.swimmer
            deep = bool(life.swim.deep_ponds())
            state = "уже пловец - глубокая вода ему открыта" if sw.can_swim() else "пока не пловец"
            self.numbers.setText(f"{sw.attempts} {plural(sw.attempts, 'попытка', 'попытки', 'попыток')}, {state}."
                                 + ("" if deep else " В этом мире глубокой воды нет."))
        elif self.name == "explore" and life is not None:
            self.numbers.setText(f"Изучено {life.walk.places.explored() * 100:.0f} % мира, "
                                 f"разворотов по своей воле {life.walk.turns}, осмотрелся {life.walk.looks} раз.")


class CatalogPage(Page):
    def __init__(self, window):
        super().__init__(window)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        outer.setSpacing(theme.GAP_BIG)
        outer.addWidget(label("Каталог", "heading"))
        outer.addWidget(label("Игрушки из его сундука и занятия. Чем заняться, он выбирает сам - по тому, "
                              "что ему нравится и что уже надоело. Предложить можно; согласится ли - решит он.",
                              "hint", wrap=True))
        top = card()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
        tl.setSpacing(theme.GAP_BIG)
        self.needs = NeedsView()
        self.needs.setMinimumWidth(260)
        tl.addWidget(self.needs, 2)
        self.pack = label("", "muted", wrap=True)
        tl.addWidget(self.pack, 3)
        outer.addWidget(top)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("page")
        self.grid = QGridLayout(inner)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(theme.GAP_BIG)
        self.grid.setVerticalSpacing(theme.GAP_BIG)
        self.cards = {}
        items = [(t.name, t.title, "", t) for t in registry.TOYS.values()]
        items += [(n, ACT_TITLES[n], ACT_NOTES[n], None) for n in ACT_TITLES]
        notes = {"guitar": "Учит мелодии из песенника и свои, играет своё. Сеть-музыкант, звук струн - синтез."}
        for i, (name, title, note, toy) in enumerate(items):
            c = ActivityCard(self, name, title, note or notes.get(name, ""), toy)
            self.cards[name] = c
            self.grid.addWidget(c, i // 2, i % 2)
        self.grid.setRowStretch(len(items) // 2 + 1, 1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        self.status = label("", "hint", wrap=True)
        outer.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.apply_theme(window.settings.theme)

    def life(self):
        return self.window_.observe.life

    def on_show(self):
        self.refresh()
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def apply_theme(self, name):
        colors = theme.palette(name)
        self.needs.set_theme(name)
        for n, c in self.cards.items():
            c.icon.setPixmap(toy_icon(n, colors))

    def refresh(self):
        life = self.life()
        fav = None
        if life is not None:
            self.needs.set_state(life.choice.mood, life.needs)
            b = life.backpack
            self.pack.setText(f"Рюкзак: еды {b.food} {plural(b.food, 'порция', 'порции', 'порций')}, фляга "
                              f"{round(b.flask * 100)} %, лакомств для собаки {b.treats}. Пополняется у сундука. "
                              "Есть, пить и отдыхать он идёт сам, когда нужда сильнее, чем тянет играть.")
            acts = {n: a.liking for n, a in life.choice.acts.items() if n in self.cards}
            fav = max(acts, key=acts.get) if acts else None
        for c in self.cards.values():
            c.refresh(life, fav)

    def offer(self, name):
        life = self.life()
        if life is None:
            return None
        res = life.offer(name)
        if res is None:
            return None
        took, events = res
        self.window_.observe.handle(life, events)
        self.status.setText("Взял - играет." if took else "Не взял: сейчас ему интереснее другое. Лежит рядом.")
        self.refresh()
        return took

