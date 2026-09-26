"""Чат над миром: фразы человечка, как в мессенджере - время, пузырь с текстом, прокрутка.
Новые снизу, лента сама едет вниз; хранится не больше MAX_MESSAGES."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

MAX_MESSAGES = 120


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:d}:{s % 60:02d}"


class ChatView(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QWidget()
        self.inner.setObjectName("chatInner")
        self.col = QVBoxLayout(self.inner)
        self.col.setContentsMargins(12, 8, 12, 8)
        self.col.setSpacing(6)
        self.col.addStretch(1)
        self.setWidget(self.inner)
        self.rows = []
        self.placeholder = QLabel("Здесь появятся его фразы")
        self.placeholder.setObjectName("hint")
        self.col.addWidget(self.placeholder)

    def count(self):
        return len(self.rows)

    def texts(self):
        return [row.findChild(QLabel, "bubble").text() for row in self.rows]

    def clear_messages(self, note=None):
        for row in self.rows:
            row.deleteLater()
        self.rows = []
        self.placeholder.setText(note or "Здесь появятся его фразы")
        self.placeholder.show()

    def add(self, seconds, text, who=""):
        self.placeholder.hide()
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        bubble = QLabel(text[:1].upper() + text[1:] if text else text)
        bubble.setObjectName("bubble")
        bubble.setWordWrap(True)
        stamp = QLabel(f"{who}  {clock(seconds)}" if who else clock(seconds))
        stamp.setObjectName("bubbleTime")
        line.addWidget(bubble, 0, Qt.AlignmentFlag.AlignLeft)
        line.addWidget(stamp, 0, Qt.AlignmentFlag.AlignBottom)
        line.addStretch(1)
        self.col.addWidget(row)
        self.rows.append(row)
        while len(self.rows) > MAX_MESSAGES:
            self.rows.pop(0).deleteLater()
        QTimer.singleShot(0, self.to_bottom)
        return row

    def to_bottom(self):
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
