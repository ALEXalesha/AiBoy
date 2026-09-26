"""Учитель речи - только для обучения (train/speech.py), в игру не попадает.

По сводке состояния мозга (brain/state.py) пишет фразу из шаблонов: о чём сейчас важнее
сказать (упал, летит, рядом сущность, видно холм или яму, хочется прыгнуть, скучно или
интересно, ночь), с согласованием прилагательного и существительного по роду. Какую из
подходящих тем и какой шаблон взять - случайно, с весами. Сеть речи учится повторять эти
фразы по тому же состоянию - и в игре говорит уже сама.
"""
import math

import numpy as np

from brain.state import IDX
from world.generate import KINDS

# существительное: им. п., род, вин. п., дат. п.
NOUNS = {
    "tree": ("дерево", "n", "дерево", "дереву"),
    "bush": ("куст", "m", "куст", "кусту"),
    "stone": ("камень", "m", "камень", "камню"),
    "flower": ("цветок", "m", "цветок", "цветку"),
    "mushroom": ("гриб", "m", "гриб", "грибу"),
    "crystal": ("кристалл", "m", "кристалл", "кристаллу"),
}
# прилагательное цвета: мужской и средний род
COLORS = {
    "red": ("красный", "красное"), "orange": ("оранжевый", "оранжевое"), "yellow": ("жёлтый", "жёлтое"),
    "green": ("зелёный", "зелёное"), "cyan": ("голубой", "голубое"), "blue": ("синий", "синее"),
    "purple": ("фиолетовый", "фиолетовое"), "pink": ("розовый", "розовое"), "white": ("белый", "белое"),
    "grey": ("серый", "серое"), "black": ("чёрный", "чёрное"),
}
HUES = ((0.04, "red"), (0.11, "orange"), (0.19, "yellow"), (0.45, "green"), (0.54, "cyan"),
        (0.72, "blue"), (0.83, "purple"), (0.95, "pink"), (1.01, "red"))
SIDE = {1: "справа", -1: "слева"}
WAY = {1: "направо", -1: "налево"}

FALL = ("ой!", "ай!", "упал...", "ой, больно", "ничего, встаю", "бывает", "земля твёрдая", "ух, упал")
FLY = ("лечу!", "уи-и!", "высоко!", "прыг!")
JUMP = ("попробую прыгнуть", "а если прыгнуть?", "прыгну!")
CURIOUS = ("интересно...", "ух ты!", "что это?", "как тут всё странно", "всё новое!")
BORED = ("скучно", "хм.", "тут я уже был", "всё знакомо", "пойду дальше")
NIGHT = ("темно...", "звёзды!", "ночь. тихо", "луна светит")
FLAT = ("ровное место", "иду дальше", "тут ровно")
FAST = ("бегу!", "быстрее!")
IDLE = ("хм...", "ла-ла-ла", "я тут", "хорошо!", "привет, мир!")
DEEP = ("не упасть бы",)
UP = ("надо забраться повыше",)


def color_name(h, s, v):
    if s < 0.2:
        return "white" if v > 0.8 else ("black" if v < 0.3 else "grey")
    for edge, name in HUES:
        if h < edge:
            return name
    return "red"


def _entity(state):
    kind = next((k for k in KINDS if state[IDX["kind_" + k]] > 0.5), None)
    if kind is None or state[IDX["entity"]] < 0.5:
        return None
    hue = (math.atan2(state[IDX["hue_sin"]], state[IDX["hue_cos"]]) / (2 * math.pi)) % 1.0
    return kind, color_name(hue, state[IDX["sat"]], state[IDX["val"]])


def _side(v):
    return 1 if v >= 0 else -1


def entity_phrases(kind, color, side, near):
    noun, gender, acc, dat = NOUNS[kind]
    adj = COLORS[color][0 if gender == "m" else 1]
    neut = COLORS[color][1]
    if near:
        return [f"тут что-то {neut}", f"{adj} {noun}!", f"какой {adj} {noun}" if gender == "m"
                else f"какое {adj} {noun}", f"потрогаю {acc}", f"это {noun}?", f"вот это {noun}!"]
    return [f"вижу {noun} {SIDE[side]}", f"{SIDE[side]} что-то {neut}", f"пойду к {dat}",
            f"там {adj} {noun}"]


def topics(state):
    """[(вес, [фразы])] - что можно сказать в этом состоянии."""
    s = lambda name: float(state[IDX[name]])  # noqa: E731
    out = [(0.7, list(IDLE))]
    if s("fell") > 0.5:
        out.append((6.0, list(FALL)))
    if s("airborne") > 0.5:
        out.append((3.0, list(FLY)))
    ent = _entity(state)
    if ent:
        near = s("entity_dist") < 0.25
        out.append((4.0 if near else 3.0, entity_phrases(ent[0], ent[1], _side(s("entity_side")), near)))
    for side, name in ((1, "right"), (-1, "left")):
        if s("hill_" + name) > 0.4:
            out.append((2.5, [f"вижу высокий холм {SIDE[side]}", f"{SIDE[side]} гора", f"холм {SIDE[side]}!"]
                        + list(UP)))
        if s("pit_" + name) > 0.4:
            out.append((2.5, [f"осторожно, яма {SIDE[side]}", f"{SIDE[side]} обрыв"] + list(DEEP)))
    if s("want_jump") > 0.3:
        out.append((2.0, list(JUMP)))
    if abs(s("want_dir")) > 0.5:
        d = _side(s("want_dir"))
        out.append((1.5, [f"пойду {WAY[d]}", f"что там {SIDE[d]}?"]))
    if s("curiosity") > 0.6:
        out.append((2.0, list(CURIOUS)))
    if s("curiosity") < 0.25:
        out.append((1.5, list(BORED)))
    if s("night") > 0.5:
        out.append((1.5, list(NIGHT)))
    if s("flat") > 0.7:
        out.append((1.0, list(FLAT)))
    if s("speed") > 0.6:
        out.append((1.5, list(FAST)))
    return out


def phrase(state, rng):
    ts = topics(state)
    w = np.array([t[0] for t in ts])
    choice = ts[int(rng.choice(len(ts), p=w / w.sum()))][1]
    return choice[int(rng.integers(len(choice)))]


def words(text):
    return [w for w in "".join(c if c.isalpha() or c == "-" else " " for c in text).split() if w.strip("-")]


def vocabulary():
    """Все слова, которые учитель может сказать."""
    texts = list(FALL + FLY + JUMP + CURIOUS + BORED + NIGHT + FLAT + FAST + IDLE + DEEP + UP)
    for kind in KINDS:
        for color in COLORS:
            for side in (1, -1):
                for near in (True, False):
                    texts += entity_phrases(kind, color, side, near)
    for side in (1, -1):
        texts += [f"вижу высокий холм {SIDE[side]}", f"{SIDE[side]} гора", f"холм {SIDE[side]}!",
                  f"осторожно, яма {SIDE[side]}", f"{SIDE[side]} обрыв", f"пойду {WAY[side]}",
                  f"что там {SIDE[side]}?"]
    return {w for t in texts for w in words(t)}
