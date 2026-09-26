"""Общие кирпичики окна: подпись, карточка, кнопка, ряд переключателей, страница."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget


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
