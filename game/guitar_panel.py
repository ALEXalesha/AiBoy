"""Панель гитары в наблюдении, пока он играет: гриф с пальцами, ноты попытки строкой, какую
мелодию учит, как вышла прошлая попытка, и 👍/👎 владельца.

Гриф нарисован как вид сверху: шесть струн (внизу - самая низкая, как в табулатуре),
лады 0-12. Где палец - точка на ладу последней сыгранной ноты; струна, по которой только что
ударил, светится. Под грифом - строка нот попытки по времени (выше - выше звук), у мелодии -
контуры нот, которые надо сыграть; вертикальная черта - где он сейчас. В режиме «Руками»
промахи - пустые красные.
"""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from game import theme
from game.ui import button, card, label
from music import guitar


class NeckView(QWidget):
    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self.attempt = None
        self.t = 0.0
        self.setMinimumHeight(104)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(600, 110)

    def set_attempt(self, attempt):
        self.attempt = attempt
        self.t = 0.0
        self.update()

    def set_time(self, t):
        self.t = t
        self.update()

    def current(self):
        """Последняя сыгранная к этому моменту нота и прошло ли с удара меньше 0.2 с."""
        a = self.attempt
        if a is None:
            return None, False
        last = None
        for (step, s, f, force, d), snd in zip(a.intended, a.sounded):
            on = step * a.step_sec
            if on <= self.t:
                last = (s, f, snd, self.t - on < 0.2)
        if last is None:
            return None, False
        return last, last[3]

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = self.colors
        w, h = self.width(), self.height()
        neck = QRectF(28, 4, w - 36, 52)
        # лады: положения как у настоящей гитары, но растянуты на ширину
        def fret_x(k):
            full = 1.0 - 2.0 ** (-12 / 12.0)
            return neck.left() + neck.width() * (1.0 - 2.0 ** (-k / 12.0)) / full
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#6b4a2f"))
        p.drawRoundedRect(neck, 4, 4)
        p.setPen(QPen(QColor("#d9d2c5"), 2))
        for k in range(1, guitar.FRETS):
            x = fret_x(k)
            p.drawLine(QPointF(x, neck.top()), QPointF(x, neck.bottom()))
        p.setPen(QPen(QColor("#f0ead6"), 4))
        p.drawLine(QPointF(neck.left(), neck.top()), QPointF(neck.left(), neck.bottom()))
        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        for k in (3, 5, 7, 9, 12):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 60))
            cx = (fret_x(k - 1) + fret_x(k)) / 2
            p.drawEllipse(QPointF(cx, neck.center().y()), 3, 3)
        last, fresh = self.current()

        def string_y(s):
            return neck.bottom() - 6 - s * (neck.height() - 12) / (guitar.STRINGS - 1)

        for s in range(guitar.STRINGS):
            glow = fresh and last is not None and last[0] == s
            pen = QPen(QColor(col["warm"]) if glow else QColor("#d8d8d8"), 1.0 + (guitar.STRINGS - s) * 0.35
                       + (1.5 if glow else 0))
            p.setPen(pen)
            y = string_y(s)
            p.drawLine(QPointF(neck.left() - 20, y), QPointF(neck.right(), y))
            p.setPen(QColor(col["muted"]))
            p.drawText(QRectF(0, y - 7, 18, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       "EADGBE"[s])
        if last is not None:
            s, f, snd, _ = last
            x = neck.left() - 10 if f == 0 else (fret_x(f - 1) + fret_x(f)) / 2
            dot = QColor(col["accent"]) if snd == 1 else QColor(col["bad"])
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.setBrush(dot if snd else Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x, string_y(s)), 7, 7)
        # строка нот попытки
        roll = QRectF(28, neck.bottom() + 6, w - 36, h - neck.bottom() - 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(col["surface2"]))
        p.drawRoundedRect(roll, 4, 4)
        a = self.attempt
        if a is None:
            p.end()
            return
        lo, hi = guitar.LOW, guitar.HIGH

        def rect_for(step, dur, pitch):
            x = roll.left() + roll.width() * step / a.steps
            ww = max(3.0, roll.width() * dur / a.steps - 1)
            y = roll.bottom() - 3 - (roll.height() - 8) * (pitch - lo) / (hi - lo)
            return QRectF(x, y - 2, ww, 4)

        if a.phrase is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(col["muted"]), 1))
            for n in a.phrase.notes:
                p.drawRect(rect_for(n.onset, n.dur, n.pitch))
        for (step, s, f, force, d), snd in zip(a.intended, a.sounded):
            r = rect_for(step, d, guitar.midi(s, f))
            if snd == 1:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(col["accent"]))
            else:
                p.setPen(QPen(QColor(col["bad"]), 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
        x = roll.left() + roll.width() * min(1.0, self.t / a.duration)
        p.setPen(QPen(QColor(col["warm"]), 2))
        p.drawLine(QPointF(x, roll.top()), QPointF(x, roll.bottom()))
        p.end()


class GuitarPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        box = card()
        outer.addWidget(box)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
        lay.setSpacing(theme.GAP_SMALL)
        top = QHBoxLayout()
        top.setSpacing(theme.GAP)
        words = QVBoxLayout()
        words.setSpacing(2)
        self.title = label("", "section", wrap=True)
        self.result = label("", "hint", wrap=True)
        words.addWidget(self.title)
        words.addWidget(self.result)
        top.addLayout(words, 1)
        self.like = button("👍 Нравится", None, lambda: self.rate(1))
        self.dislike = button("👎 Не нравится", None, lambda: self.rate(-1))
        top.addWidget(self.like, 0, Qt.AlignmentFlag.AlignTop)
        top.addWidget(self.dislike, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        self.neck = NeckView(window.colors())
        lay.addWidget(self.neck)

    def rate(self, rating):
        life = self.window_.observe.life
        if life is None:
            return
        entry = life.rate(rating)
        word = "👍" if rating > 0 else "👎"
        if entry is None and life.attempt is not None:
            self.result.setText(f"Оценка {word} войдёт в эту попытку")
        elif entry is not None:
            self.result.setText(f"Попытка {entry['id']}: оценка {word} учтена")
            self.window_.observe.after_rating(rating)

    def set_attempt(self, attempt):
        if attempt.tune is None:
            self.title.setText("Свободная игра - играет своё")
        else:
            self.title.setText(f"Учит «{attempt.tune}», фраза {attempt.phrase.index + 1}")
        self.neck.set_attempt(attempt)

    def set_result(self, entry):
        parts = []
        if entry.get("A") is not None:
            parts.append(f"похоже на мелодию {entry['A']:.2f}")
        parts.append(f"приятно {entry['B']:+.2f}")
        if entry.get("rating") is not None:
            parts.append("👍" if entry["rating"] > 0 else "👎")
        elif entry.get("taste") is not None:
            parts.append(f"вкус угадывает {entry['taste']:+.2f}")
        self.result.setText(f"Попытка {entry['id']}: " + ", ".join(parts))

    def apply_theme(self, name):
        self.neck.colors = theme.palette(name)
        self.neck.update()
