"""Экран «Научить мелодии»: две октавы клавиш (мышь и клавиатура компьютера), запись,
стоп, слушать, название, сохранить. Сохранённая мелодия встаёт в песенник рядом со
встроенными - и он может её учить, как любую другую.

Запись одноголосная: новая клавиша заканчивает прежнюю ноту. Время нажатий переводится в
восьмые по выбранному темпу: длительность - до отпускания (или до следующего нажатия),
паузы между нотами тоже записываются. Тишина до первой ноты не считается.
"""
import time

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea, QSizePolicy,
                               QSpinBox, QVBoxLayout, QWidget)

from game import theme
from game.ui import Page, button, card, label
from music import guitar
from music.guitar import Note
from music.songbook import Tune, in_range

LOW = 60                                   # до первой октавы
KEYS = 24
_ROW1 = (Qt.Key.Key_Z, Qt.Key.Key_S, Qt.Key.Key_X, Qt.Key.Key_D, Qt.Key.Key_C, Qt.Key.Key_V, Qt.Key.Key_G,
         Qt.Key.Key_B, Qt.Key.Key_H, Qt.Key.Key_N, Qt.Key.Key_J, Qt.Key.Key_M)
_ROW2 = (Qt.Key.Key_Q, Qt.Key.Key_2, Qt.Key.Key_W, Qt.Key.Key_3, Qt.Key.Key_E, Qt.Key.Key_R, Qt.Key.Key_5,
         Qt.Key.Key_T, Qt.Key.Key_6, Qt.Key.Key_Y, Qt.Key.Key_7, Qt.Key.Key_U)
KEYMAP = {k: LOW + i for i, k in enumerate(_ROW1)}
KEYMAP.update({k: LOW + 12 + i for i, k in enumerate(_ROW2)})
BLACK = {1, 3, 6, 8, 10}


class Recorder:
    """Нажатия и отпускания со временем -> ноты в восьмых по темпу."""

    def __init__(self, tempo=100):
        self.tempo = tempo
        self.events = []            # [(высота или None, секунд)]
        self.current = None         # (высота, с какого момента)
        self.last_end = None

    @property
    def eighth(self):
        return 60.0 / self.tempo / 2.0

    def start(self, t):
        self.events, self.current, self.last_end = [], None, None

    def _close(self, t):
        pitch, since = self.current
        self.events.append((pitch, t - since))
        self.current = None
        self.last_end = t

    def press(self, pitch, t):
        if self.current is not None:
            self._close(t)
        elif self.last_end is not None and t - self.last_end >= 0.5 * self.eighth:
            self.events.append((None, t - self.last_end))
        self.current = (pitch, t)

    def release(self, pitch, t):
        if self.current is not None and self.current[0] == pitch:
            self._close(t)

    def stop(self, t):
        if self.current is not None:
            self._close(t)
        return tuple((p, max(1, int(round(sec / self.eighth)))) for p, sec in self.events)


def note_name(pitch):
    return "пауза" if pitch is None else guitar.name(pitch)


class Piano(QWidget):
    pressed = Signal(int)
    released = Signal(int)

    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self.down = set()
        self._mouse = None
        self.setMinimumHeight(150)
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(700, 170)

    def _white_width(self):
        return self.width() / 14.0

    def key_rect(self, pitch):
        k = pitch - LOW
        octave, note = divmod(k, 12)
        whites_before = octave * 7 + sum(1 for i in range(note) if i not in BLACK)
        w = self._white_width()
        if note in BLACK:
            return QRectF(whites_before * w - w * 0.3, 0, w * 0.6, self.height() * 0.62)
        return QRectF(whites_before * w, 0, w, self.height())

    def pitch_at(self, pos):
        for p in range(LOW, LOW + KEYS):                    # сначала чёрные - они сверху
            if (p - LOW) % 12 in BLACK and self.key_rect(p).contains(pos):
                return p
        for p in range(LOW, LOW + KEYS):
            if (p - LOW) % 12 not in BLACK and self.key_rect(p).contains(pos):
                return p
        return None

    def mousePressEvent(self, event):
        p = self.pitch_at(event.position())
        if p is not None:
            self._mouse = p
            self.down.add(p)
            self.pressed.emit(p)
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mouse is not None:
            p, self._mouse = self._mouse, None
            self.down.discard(p)
            self.released.emit(p)
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = self.colors
        font = QFont(self.font())
        font.setPixelSize(11)
        p.setFont(font)
        for black in (False, True):
            for pitch in range(LOW, LOW + KEYS):
                if ((pitch - LOW) % 12 in BLACK) != black:
                    continue
                r = self.key_rect(pitch)
                on = pitch in self.down
                if black:
                    fill = QColor(col["accent"]) if on else QColor("#1b1f27")
                else:
                    fill = QColor(col["accent"]) if on else QColor("#f4f5f8")
                p.setPen(QPen(QColor(col["border"]), 1))
                p.setBrush(fill)
                p.drawRoundedRect(r.adjusted(1, 0, -1, -1), 4, 4)
                if not black and (pitch - LOW) % 12 == 0:
                    p.setPen(QColor("#636b7c"))
                    p.drawText(r.adjusted(0, 0, 0, -6), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                               guitar.name(pitch))
        p.end()


def tune_notes(notes, tempo):
    """Мелодия -> ноты гитары для «послушать»: каждая высота на самом нижнем ладу."""
    tune = in_range(Tune("x", tuple(notes), tempo, 8, builtin=False))
    eighth = 60.0 / tempo / 2.0
    out, t = [], 0
    for pitch, d in tune.notes:
        if pitch is not None:
            s, f = min(guitar.positions(pitch), key=lambda sf: sf[1])
            out.append(Note(s, f, t * eighth, 1, d * eighth))
        t += d
    return out


class KeyboardPage(Page):
    def __init__(self, window):
        super().__init__(window)
        self.clock = time.monotonic
        self.recorder = Recorder()
        self.recording = False
        self.notes = ()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        col.addWidget(label("Научить мелодии", "heading"))
        col.addWidget(label("Сыграйте мелодию мышью или клавишами компьютера (Z-M - первая октава, "
                            "Q-U - вторая, чёрные клавиши - ряд выше), дайте ей название и сохраните: "
                            "она встанет в песенник, и он будет её учить.", "hint", wrap=True))
        box = card()
        grid = QVBoxLayout(box)
        grid.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        grid.setSpacing(theme.GAP)
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        row.addWidget(label("Темп", "muted"))
        self.tempo = QSpinBox()
        self.tempo.setRange(70, 120)
        self.tempo.setValue(100)
        self.tempo.setSuffix(" в минуту")
        row.addWidget(self.tempo)
        row.addWidget(label("Размер", "muted"))
        self.meter = QComboBox()
        self.meter.addItem("4/4", 8)
        self.meter.addItem("3/4", 6)
        row.addWidget(self.meter)
        row.addStretch(1)
        grid.addLayout(row)
        self.piano = Piano(window.colors())
        self.piano.pressed.connect(self.on_press)
        self.piano.released.connect(self.on_release)
        grid.addWidget(self.piano)
        buttons = QGridLayout()
        buttons.setHorizontalSpacing(theme.GAP)
        buttons.setVerticalSpacing(theme.GAP)
        self.record_button = button("● Запись", "primary", self.start_recording)
        self.stop_button = button("■ Стоп", None, self.stop_recording)
        self.listen_button = button("▶ Слушать", None, self.listen)
        for i, b in enumerate((self.record_button, self.stop_button, self.listen_button)):
            buttons.addWidget(b, 0, i)
        grid.addLayout(buttons)
        self.shown = label("Пока ничего не записано", "muted", wrap=True)
        grid.addWidget(self.shown)
        name_row = QHBoxLayout()
        name_row.setSpacing(theme.GAP)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Название мелодии")
        self.name.setMaxLength(40)
        name_row.addWidget(self.name, 1)
        self.save_button = button("Сохранить в песенник", "primary", self.save)
        name_row.addWidget(self.save_button)
        grid.addLayout(name_row)
        self.status = label("", "hint", wrap=True)
        grid.addWidget(self.status)
        col.addWidget(box)
        self.own = label("", "hint", wrap=True)
        col.addWidget(self.own)
        col.addStretch(1)

    def on_show(self):
        self.setFocus()
        self.refresh_own()

    def refresh_own(self):
        own = [t.name for t in self.window_.songbook.own]
        self.own.setText("Ваши мелодии: " + ", ".join(own) if own else "Своих мелодий пока нет.")

    def apply_theme(self, name):
        self.piano.colors = theme.palette(name)
        self.piano.update()

    # --- запись ---
    def start_recording(self):
        self.recorder = Recorder(self.tempo.value())
        self.recorder.start(self.clock())
        self.recording = True
        self.status.setText("Идёт запись...")
        self.setFocus()

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.notes = self.recorder.stop(self.clock())
        self.shown.setText(" ".join(note_name(p) + (f"×{d}" if d != 1 else "") for p, d in self.notes)
                           or "Ничего не сыграно")
        self.status.setText(f"Записано нот: {sum(1 for p, _ in self.notes if p is not None)}")

    def on_press(self, pitch):
        w = self.window_
        s, f = min(guitar.positions(in_range(Tune("k", ((pitch, 1),), 100, 8)).notes[0][0]), key=lambda sf: sf[1])
        w.audio.play(w.bank.note_audio(Note(s, f, 0.0, 1, 0.6)) * 0.4, f"клавиша {pitch}")
        if self.recording:
            self.recorder.press(pitch, self.clock())

    def on_release(self, pitch):
        if self.recording:
            self.recorder.release(pitch, self.clock())

    def keyPressEvent(self, event):
        pitch = KEYMAP.get(event.key())
        if pitch is not None and not event.isAutoRepeat():
            self.piano.down.add(pitch)
            self.piano.update()
            self.on_press(pitch)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        pitch = KEYMAP.get(event.key())
        if pitch is not None and not event.isAutoRepeat():
            self.piano.down.discard(pitch)
            self.piano.update()
            self.on_release(pitch)
            return
        super().keyReleaseEvent(event)

    # --- слушать и сохранить ---
    def listen(self):
        if not self.notes:
            self.status.setText("Сначала запишите мелодию")
            return
        w = self.window_
        name = self.name.text().strip() or "новая"
        w.audio.play(w.bank.render(tune_notes(self.notes, self.tempo.value())), f"мелодия: {name}")

    def save(self):
        name = self.name.text().strip()
        if not self.notes:
            self.status.setText("Сначала запишите мелодию")
            return
        if not name:
            self.status.setText("Нужно название")
            return
        tune = Tune(name, tuple(self.notes), self.tempo.value(), self.meter.currentData(), builtin=False)
        try:
            self.window_.songbook.add(tune)
        except ValueError:
            self.status.setText("Так называется встроенная мелодия - выберите другое название")
            return
        self.window_.songbook.save()
        self.status.setText(f"«{name}» - в песеннике")
        self.refresh_own()
