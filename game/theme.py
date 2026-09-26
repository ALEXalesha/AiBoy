"""Две темы окна - тёмная и светлая: цвета панелей и таблица стилей QSS. Мир рисуется
своими цветами (их придумала сеть мира), тема - только рамка вокруг."""

PALETTES = {
    "dark": {
        "bg": "#111419", "surface": "#1a1e26", "surface2": "#232935", "border": "#2e3542",
        "text": "#e8ebf0", "muted": "#8d96a8", "accent": "#5fb8a8", "accent_text": "#0d1a17",
        "warm": "#f2a65a", "good": "#3ecf8e", "bad": "#ff5d6c", "grid": "#262c38",
        "bubble": "#232935", "bubble_text": "#e8ebf0",
    },
    "light": {
        "bg": "#f4f5f8", "surface": "#ffffff", "surface2": "#eceff4", "border": "#d9dde6",
        "text": "#1b1f27", "muted": "#636b7c", "accent": "#2f8f81", "accent_text": "#ffffff",
        "warm": "#d9822b", "good": "#1d9c69", "bad": "#d8404f", "grid": "#e3e7ee",
        "bubble": "#eef1f6", "bubble_text": "#1b1f27",
    },
}

THEME_NAMES = {"dark": "Тёмная", "light": "Светлая"}


def palette(name):
    return PALETTES.get(name, PALETTES["dark"])


def qss(name):
    p = palette(name)
    return f"""
* {{
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 14px;
    color: {p['text']};
}}
QMainWindow, QWidget#page, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget#page,
QWidget#galleryGrid, QWidget#chatInner {{
    background: {p['bg']};
}}
QWidget#chatInner {{ background: {p['surface']}; }}
QScrollArea {{ border: none; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QLabel#heading {{ font-size: 24px; font-weight: 700; }}
QLabel#section {{ font-size: 16px; font-weight: 600; }}
QLabel#worldName {{ font-size: 20px; font-weight: 700; }}
QLabel#muted, QLabel#hint {{ color: {p['muted']}; }}
QLabel#hint {{ font-size: 12px; }}
QLabel#bigNumber {{ font-size: 26px; font-weight: 700; }}
QLabel#value {{ font-weight: 600; }}
QLabel#bubble {{
    background: {p['bubble']};
    color: {p['bubble_text']};
    border-radius: 12px;
    padding: 6px 12px;
}}
QLabel#bubbleTime {{ color: {p['muted']}; font-size: 11px; }}
QFrame#card {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 14px;
}}
QFrame#thumb {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 12px;
}}
QFrame#thumb:hover {{ border-color: {p['accent']}; }}
QPushButton {{
    background: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 10px;
    padding: 8px 14px;
}}
QPushButton:hover {{ border-color: {p['accent']}; }}
QPushButton:pressed {{ background: {p['border']}; }}
QPushButton:focus {{ outline: none; }}
QPushButton#primary {{
    background: {p['accent']};
    color: {p['accent_text']};
    border: 1px solid {p['accent']};
    font-weight: 600;
}}
QPushButton#danger:hover {{ border-color: {p['bad']}; color: {p['bad']}; }}
QPushButton#tab {{
    border: none;
    background: transparent;
    padding: 8px 16px;
    border-radius: 9px;
    color: {p['muted']};
    font-weight: 600;
}}
QPushButton#tab:hover {{ color: {p['text']}; }}
QPushButton#tab:checked {{ background: {p['surface2']}; color: {p['text']}; }}
QPushButton#seg {{
    border-radius: 8px;
    padding: 6px 8px;
    background: {p['surface2']};
}}
QPushButton#seg:checked {{
    background: {p['accent']};
    color: {p['accent_text']};
    border-color: {p['accent']};
    font-weight: 600;
}}
QComboBox {{
    background: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    padding: 6px 10px;
    min-width: 150px;
}}
QComboBox:hover {{ border-color: {p['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    selection-background-color: {p['accent']};
    selection-color: {p['accent_text']};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {p['surface2']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {p['accent']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {p['text']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {p['border']};
    background: {p['surface2']};
}}
QCheckBox::indicator:checked {{
    background: {p['accent']};
    border-color: {p['accent']};
    image: none;
}}
QTableWidget {{
    background: {p['surface']};
    border: none;
    gridline-color: {p['grid']};
    selection-background-color: {p['surface2']};
    selection-color: {p['text']};
}}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{
    background: {p['surface']};
    color: {p['muted']};
    border: none;
    border-bottom: 1px solid {p['border']};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {p['surface']}; border: none; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{
    background: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
}}
"""
