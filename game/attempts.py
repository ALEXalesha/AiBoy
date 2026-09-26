"""Экран «Попытки»: журнал гитары текущего человечка - список (дата, мелодия, оценка, 👍/👎),
«Слушать», «Сравнить с первой по этой мелодии» (первая, потом выбранная), оценка выбранной
и график: как росла похожесть на мелодию (или приятность в свободной игре) по попыткам.

Звук каждый раз синтезируется заново по нотам из журнала.
"""
import time

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QSizePolicy, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from game import theme
from game.ui import Page, button, card, label
from music.guitar import Note

FREE = "Свободная игра"


def notes_of(entry):
    step = 60.0 / entry.get("tempo", 100) / 2.0
    out = []
    for t, s, f, force, d, snd in entry["notes"]:
        if snd == 1:
            out.append(Note(s, f, t * step, force, d * step))
        elif snd == 2:
            out.append(Note(s, 0, t * step, 0, min(d, 1) * step, muted=True))
    return out, entry.get("steps", 32) * step


def score_of(entry):
    return entry["A"] if entry.get("A") is not None else entry.get("B", 0.0)


class Progress(QWidget):
    """Точки попыток и скользящее среднее по 10."""

    def __init__(self, colors):
        super().__init__()
        self.colors = colors
        self.points = []
        self.what = ""
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, points, what):
        self.points, self.what = list(points), what
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = self.colors
        font = QFont(self.font())
        font.setPixelSize(12)
        p.setFont(font)
        plot = QRectF(44, 10, self.width() - 60, self.height() - 40)
        lo, hi = (0.0, 1.0) if self.what == "A" else (-1.0, 1.0)
        for i in range(5):
            y = plot.bottom() - plot.height() * i / 4
            p.setPen(QPen(QColor(col["grid"]), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(col["muted"]))
            p.drawText(QRectF(0, y - 8, plot.left() - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{lo + (hi - lo) * i / 4:.2f}")
        if len(self.points) < 2:
            p.setPen(QColor(col["muted"]))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Попыток пока мало")
            p.end()
            return
        n = len(self.points)

        def at(i, v):
            return QPointF(plot.left() + plot.width() * i / (n - 1),
                           plot.bottom() - plot.height() * (min(max(v, lo), hi) - lo) / (hi - lo))

        p.setPen(Qt.PenStyle.NoPen)
        dot = QColor(col["accent"])
        dot.setAlphaF(0.35)
        p.setBrush(dot)
        for i, (_, v) in enumerate(self.points):
            p.drawEllipse(at(i, v), 2.5, 2.5)
        vals = [v for _, v in self.points]
        path = QPainterPath()
        for i in range(n):
            w = vals[max(0, i - 9):i + 1]
            pt = at(i, sum(w) / len(w))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        pen = QPen(QColor(col["accent"]), 2.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(QColor(col["muted"]))
        p.drawText(QRectF(plot.left(), plot.bottom() + 6, plot.width(), 16), Qt.AlignmentFlag.AlignCenter,
                   f"попытки 1-{n}; линия - среднее по 10")
        p.end()


class AttemptsPage(Page):
    def __init__(self, window):
        super().__init__(window)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 16, 24, 16)
        outer.setSpacing(theme.GAP_BIG)
        top = QHBoxLayout()
        top.addWidget(label("Попытки", "heading"))
        top.addStretch(1)
        self.who = label("", "muted")
        top.addWidget(self.who)
        outer.addLayout(top)
        self.note = label("Каждая попытка сыграть на гитаре: ноты, оценка и ваша 👍/👎. Звук синтезируется "
                          "заново по нотам.", "hint", wrap=True)
        outer.addWidget(self.note)
        body = QHBoxLayout()
        body.setSpacing(theme.GAP_BIG)
        left = card()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        ll.setSpacing(theme.GAP)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["№", "Когда", "Мелодия", "Оценка", "Вы"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        ll.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        self.listen_button = button("▶ Слушать", "primary", self.listen)
        self.compare_button = button("Сравнить с первой", None, self.compare)
        self.like = button("👍", None, lambda: self.rate(1))
        self.dislike = button("👎", None, lambda: self.rate(-1))
        for b in (self.listen_button, self.compare_button, self.like, self.dislike):
            row.addWidget(b)
        ll.addLayout(row)
        self.status = label("", "hint", wrap=True)
        ll.addWidget(self.status)
        body.addWidget(left, 3)
        right = card()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        rl.setSpacing(theme.GAP)
        rl.addWidget(label("Как растёт", "section"))
        self.tune = QComboBox()
        self.tune.currentIndexChanged.connect(self.refresh_chart)
        rl.addWidget(self.tune)
        self.chart = Progress(window.colors())
        rl.addWidget(self.chart)
        self.chart_note = label("", "hint", wrap=True)
        rl.addWidget(self.chart_note)
        rl.addStretch(1)
        body.addWidget(right, 2)
        outer.addLayout(body, 1)
        self.rows = []

    def journal(self):
        life = self.window_.observe.life
        return None if life is None or life.guitarist is None else life.guitarist.journal

    def on_show(self):
        self.refresh()

    def apply_theme(self, name):
        self.chart.colors = theme.palette(name)
        self.chart.update()

    def refresh(self):
        j = self.journal()
        life = self.window_.observe.life
        self.who.setText("" if life is None else f"человечек {life.body.name}")
        entries = list(reversed(j.entries)) if j else []
        self.rows = entries
        self.table.setRowCount(len(entries))
        for r, e in enumerate(entries):
            when = time.strftime("%d.%m %H:%M", time.localtime(e.get("time", 0)))
            tune = e.get("tune") or FREE
            if e.get("phrase") is not None and e.get("tune"):
                tune += f", фраза {e['phrase'] + 1}"
            score = f"похоже {e['A']:.2f}" if e.get("A") is not None else f"приятно {e.get('B', 0):+.2f}"
            mark = {1: "👍", -1: "👎"}.get(e.get("rating"), "")
            for c, text in enumerate((str(e["id"]), when, tune, score, mark)):
                self.table.setItem(r, c, QTableWidgetItem(text))
        if entries and not self.table.selectedItems():
            self.table.selectRow(0)
        self.status.setText("" if entries else "Он ещё ни разу не играл: гитара лежит в мире, а берёт он её, "
                            "когда мир надоест.")
        current = self.tune.currentData()
        self.tune.blockSignals(True)
        self.tune.clear()
        names = j.tunes() if j else []
        for name in names:
            self.tune.addItem(name or FREE, name if name is not None else "")
        idx = self.tune.findData(current if current is not None else "")
        self.tune.setCurrentIndex(max(0, idx))
        self.tune.blockSignals(False)
        self.refresh_chart()

    def refresh_chart(self):
        j = self.journal()
        data = self.tune.currentData()
        tune = None if data in ("", None) else data
        if j is None or self.tune.count() == 0:
            self.chart.set_data([], "A")
            self.chart_note.setText("")
            return
        pts = j.progress(tune)
        self.chart.set_data(pts, "B" if tune is None else "A")
        if tune is None:
            self.chart_note.setText("Свободная игра: приятность (консонанс, ритм, новизна) от -1 до 1.")
        else:
            best = j.best(tune)
            self.chart_note.setText(f"Похожесть на «{tune}» от 0 до 1; лучшая - {best:.2f}.")

    def selected(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        r = rows[0].row()
        return self.rows[r] if 0 <= r < len(self.rows) else None

    def play(self, entries, label_):
        w = self.window_
        from music.string import RELEASE
        import numpy as np
        parts = []
        for e in entries:
            notes, length = notes_of(e)
            audio = w.bank.render(notes, length) if notes else np.zeros(int(length * 44100), np.float32)
            parts.append(audio)
            parts.append(np.zeros(int((0.8 + RELEASE) * 44100), np.float32))
        w.audio.play(np.concatenate(parts), label_)

    def listen(self):
        e = self.selected()
        if e is None:
            return
        self.play([e], f"попытка {e['id']}")
        self.status.setText(f"Играет попытка {e['id']}")

    def compare(self):
        e = self.selected()
        j = self.journal()
        if e is None or j is None:
            return
        first = j.first_of(e.get("tune"))
        self.play([first, e], f"сравнение {first['id']} и {e['id']}")
        self.status.setText(f"Сначала первая попытка ({first['id']}), потом эта ({e['id']})")

    def rate(self, rating):
        e = self.selected()
        life = self.window_.observe.life
        if e is None or life is None or life.guitarist is None:
            return
        g = life.guitarist
        if g.last is not None and g.last[1]["id"] == e["id"]:
            g.rate_last(rating)
        else:
            from music import taste as taste_mod
            from music.journal import played
            g.journal.rate(e["id"], rating)
            g._taste_learn(taste_mod.features(played(e), e.get("A"), e.get("steps", 32), e.get("meter", 8)), rating)
        self.window_.observe.after_rating(rating)
        self.refresh()
        self.status.setText(f"Попытка {e['id']}: {'👍' if rating > 0 else '👎'} - вкус учится на этой оценке")
