"""Иконка рисуется кодом: небо на закате, холм, солнце и человечек с пузырём речи."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap


def render(size=256):
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(0, 0, s, s), s * 0.22, s * 0.22)
    p.setClipPath(tile)
    sky = QLinearGradient(0, 0, 0, s)
    sky.setColorAt(0, QColor("#2b3a67"))
    sky.setColorAt(0.65, QColor("#f2a65a"))
    p.fillRect(QRectF(0, 0, s, s), sky)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 240, 200))
    p.drawEllipse(QPointF(s * 0.74, s * 0.34), s * 0.1, s * 0.1)
    hill = QPainterPath(QPointF(0, s * 0.78))
    hill.cubicTo(s * 0.3, s * 0.62, s * 0.6, s * 0.7, s, s * 0.66)
    hill.lineTo(s, s)
    hill.lineTo(0, s)
    hill.closeSubpath()
    p.setBrush(QColor("#2f8f81"))
    p.drawPath(hill)
    # человечек
    ink = QColor("#fdf6ec")
    w = max(1.5, s * 0.045)
    pen = QPen(ink, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    hip, neck = QPointF(s * 0.38, s * 0.62), QPointF(s * 0.39, s * 0.44)
    p.drawLine(hip, neck)
    p.drawLine(hip, QPointF(s * 0.30, s * 0.74))
    p.drawLine(hip, QPointF(s * 0.46, s * 0.73))
    p.drawLine(QPointF(s * 0.39, s * 0.47), QPointF(s * 0.28, s * 0.55))
    p.drawLine(QPointF(s * 0.39, s * 0.47), QPointF(s * 0.5, s * 0.38))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ink)
    p.drawEllipse(QPointF(s * 0.40, s * 0.37), s * 0.065, s * 0.065)
    # пузырь речи
    p.setBrush(QColor(255, 255, 255, 235))
    p.drawRoundedRect(QRectF(s * 0.5, s * 0.12, s * 0.34, s * 0.16), s * 0.08, s * 0.08)
    tail = QPainterPath(QPointF(s * 0.56, s * 0.27))
    tail.lineTo(s * 0.52, s * 0.33)
    tail.lineTo(s * 0.63, s * 0.27)
    p.drawPath(tail)
    p.setBrush(QColor("#2b3a67"))
    for k in range(3):
        p.drawEllipse(QPointF(s * (0.59 + 0.08 * k), s * 0.2), s * 0.022, s * 0.022)
    p.end()
    return image


def app_icon():
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(render(size)))
    return icon
