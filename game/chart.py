"""Графики: маленький график любопытства на панели наблюдения и кривая обучения мозга
в статистике."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from game import theme


def smooth(values, k=0.2):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else acc + (v - acc) * k
        out.append(acc)
    return out


class Sparkline(QWidget):
    """Ошибка предсказателя за последние минуты: чем выше, тем больше нового вокруг."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.colors = theme.palette("dark")
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def set_theme(self, name):
        self.colors = theme.palette(name)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(1, 4, self.width() - 2, self.height() - 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.colors["surface2"]))
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 8, 8)
        vals = smooth(self.values)
        if len(vals) >= 2:
            top = max(vals) * 1.15 or 1.0
            pts = [QPointF(r.left() + r.width() * i / (len(vals) - 1), r.bottom() - r.height() * v / top)
                   for i, v in enumerate(vals)]
            area = QPainterPath(QPointF(pts[0].x(), r.bottom()))
            for pt in pts:
                area.lineTo(pt)
            area.lineTo(pts[-1].x(), r.bottom())
            g = QLinearGradient(0, r.top(), 0, r.bottom())
            c = QColor(self.colors["warm"])
            c.setAlphaF(0.45)
            g.setColorAt(0, c)
            c.setAlphaF(0.0)
            g.setColorAt(1, c)
            p.setBrush(g)
            p.drawPath(area)
            line = QPainterPath(pts[0])
            for pt in pts[1:]:
                line.lineTo(pt)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(self.colors["warm"]), 2))
            p.drawPath(line)
        p.end()


def time_label(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} ч".replace(".", ",")
    if seconds < 120:
        return f"{round(seconds)} с"
    return f"{round(seconds / 60)} мин"


class CurveChart(QWidget):
    """Кривая обучения: средняя награда любопытства по прожитому времени."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.colors = theme.palette("dark")
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, points):
        self.points = [tuple(pt) for pt in points]
        self.update()

    def set_theme(self, name):
        self.colors = theme.palette(name)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        col = self.colors
        small = QFont(self.font())
        small.setPixelSize(12)
        p.setFont(small)
        plot = QRectF(52, 14, self.width() - 70, self.height() - 50)
        for i in range(5):
            y = plot.bottom() - plot.height() * i / 4
            p.setPen(QPen(QColor(col["grid"]), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        if len(self.points) < 2:
            p.setPen(QColor(col["muted"]))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Пока мало данных: график растёт, пока он живёт")
            p.end()
            return
        xs = [pt[0] for pt in self.points]
        ys = smooth([pt[1] for pt in self.points], 0.15)
        lo, hi = min(ys), max(ys)
        if hi - lo < 1e-6:
            hi = lo + 1e-3
        x0, x1 = xs[0], max(xs[-1], xs[0] + 1)

        def at(x, y):
            return QPointF(plot.left() + plot.width() * (x - x0) / (x1 - x0),
                           plot.bottom() - plot.height() * (y - lo) / (hi - lo))

        p.setPen(QColor(col["muted"]))
        for i in range(5):
            v = lo + (hi - lo) * i / 4
            y = plot.bottom() - plot.height() * i / 4
            p.drawText(QRectF(0, y - 8, plot.left() - 8, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{v:.3f}")
        for i in range(5):
            x = x0 + (x1 - x0) * i / 4
            pt = at(x, lo)
            p.drawText(QRectF(pt.x() - 40, plot.bottom() + 6, 80, 16), Qt.AlignmentFlag.AlignCenter, time_label(x))
        p.drawText(QRectF(plot.left(), plot.bottom() + 22, plot.width(), 16), Qt.AlignmentFlag.AlignCenter,
                   "прожито")
        path = QPainterPath(at(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(at(x, y))
        pen = QPen(QColor(col["accent"]), 2.5)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
        p.end()


class NeedsView(QWidget):
    """Настроение лицом и три нужды полосками: голод, жажда, усталость (0..1)."""

    NAMES = (("hunger", "голод"), ("thirst", "жажда"), ("tired", "усталость"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mood = 0.0
        self.values = {k: 0.0 for k, _ in self.NAMES}
        self.colors = theme.palette("dark")
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_state(self, mood, needs):
        self.mood = float(mood)
        self.values = {k: float(getattr(needs, k)) for k, _ in self.NAMES}
        self.update()

    def set_theme(self, name):
        self.colors = theme.palette(name)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.colors
        h = self.height()
        r = min(26.0, h / 2 - 4)
        cx, cy = r + 2, h / 2
        good = QColor(c["good"]) if self.mood >= 0 else QColor(c["warm"])
        p.setPen(QPen(good, 2.5))
        p.setBrush(QColor(c["surface2"]))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setBrush(good)
        p.setPen(Qt.PenStyle.NoPen)
        for dx in (-0.35, 0.35):
            p.drawEllipse(QPointF(cx + dx * r, cy - 0.25 * r), 2.6, 2.6)
        mouth = QPainterPath(QPointF(cx - 0.45 * r, cy + 0.3 * r))
        mouth.quadTo(QPointF(cx, cy + 0.3 * r + 0.5 * r * max(-1.0, min(1.0, self.mood))),
                     QPointF(cx + 0.45 * r, cy + 0.3 * r))
        p.setPen(QPen(good, 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(mouth)
        x0 = cx + r + 12
        w = self.width() - x0 - 2
        font = QFont(self.font())
        font.setPixelSize(11)
        p.setFont(font)
        for i, (k, name) in enumerate(self.NAMES):
            y = 4 + i * (h - 8) / 3
            v = self.values[k]
            p.setPen(QColor(c["muted"]))
            p.drawText(QRectF(x0, y, 72, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
            bar = QRectF(x0 + 74, y + 4, max(10.0, w - 74), 7)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(c["surface2"]))
            p.drawRoundedRect(bar, 3, 3)
            p.setBrush(QColor(c["bad"] if v > 0.75 else c["warm"] if v > 0.5 else c["accent"]))
            p.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * v, bar.height()), 3, 3)
        p.end()
