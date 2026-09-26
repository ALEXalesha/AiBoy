"""Песенник: мелодии, которые человечек учится повторять.

Мелодия - ноты (высота MIDI или None - пауза, длительность в восьмых), темп (четвертей в
минуту) и размер (восьмых в такте: 8 - 4/4, 6 - 3/4). Одноголосная. Встроенные - десять
простых мелодий в общественном достоянии; «Кузнечик» (музыка Шаинского, 1972) - под
охраной и заменён. Свои мелодии владельца (экран «Научить мелодии») лежат в songbook.json
рядом с данными игры.

Попытка - одна фраза в 4 такта. Для игры высоты переносятся октавами в диапазон гитары
(E2..E5), струну и лад выбирает сам музыкант.
"""
from dataclasses import dataclass

from game.storage import is_int, load_json, save_json
from music import guitar

MEASURES_PER_PHRASE = 4

C3, D3, E3, F3, G3, A3, B3 = 48, 50, 52, 53, 55, 57, 59
C4, D4, E4, F4, G4, A4, B4 = 60, 62, 64, 65, 67, 69, 71
C5, D5, E5, F5, G5 = 72, 74, 76, 77, 79


@dataclass(frozen=True)
class Tune:
    name: str
    notes: tuple          # ((высота или None, восьмых), ...)
    tempo: int            # четвертей в минуту
    meter: int            # восьмых в такте
    builtin: bool = True


@dataclass(frozen=True)
class Target:
    onset: int            # шаг сетки (восьмая) от начала фразы
    pitch: int
    dur: int


@dataclass(frozen=True)
class Phrase:
    tune: str
    index: int
    notes: tuple          # (Target, ...)
    steps: int
    tempo: int
    meter: int


def _q(*pitches):
    """Четверти."""
    return tuple((p, 2) for p in pitches)


BUILTIN = (
    Tune("Twinkle Twinkle", (_q(C4, C4, G4, G4, A4, A4) + ((G4, 4),) + _q(F4, F4, E4, E4, D4, D4) + ((C4, 4),)
                             + _q(G4, G4, F4, F4, E4, E4) + ((D4, 4),) + _q(G4, G4, F4, F4, E4, E4) + ((D4, 4),)
                             + _q(C4, C4, G4, G4, A4, A4) + ((G4, 4),) + _q(F4, F4, E4, E4, D4, D4) + ((C4, 4),)),
         100, 8),
    Tune("Frère Jacques", (_q(C4, D4, E4, C4) * 2 + _q(E4, F4) + ((G4, 4),) + _q(E4, F4) + ((G4, 4),)
                           + ((G4, 1), (A4, 1), (G4, 1), (F4, 1)) + _q(E4, C4)
                           + ((G4, 1), (A4, 1), (G4, 1), (F4, 1)) + _q(E4, C4)
                           + _q(C4, G3) + ((C4, 4),) + _q(C4, G3) + ((C4, 4),)), 100, 8),
    Tune("Ода к радости", (_q(E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4) + ((E4, 3), (D4, 1), (D4, 4))
                           + _q(E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4) + ((D4, 3), (C4, 1), (C4, 4))),
         100, 8),
    Tune("Mary Had a Little Lamb", (_q(E4, D4, C4, D4, E4, E4) + ((E4, 4),) + _q(D4, D4) + ((D4, 4),)
                                    + _q(E4, G4) + ((G4, 4),) + _q(E4, D4, C4, D4, E4, E4, E4, E4, D4, D4, E4, D4)
                                    + ((C4, 8),)), 100, 8),
    Tune("Jingle Bells", ((_q(E4, E4) + ((E4, 4),)) * 2 + _q(E4, G4) + ((C4, 3), (D4, 1), (E4, 8))
                          + _q(F4, F4) + ((F4, 3), (F4, 1)) + _q(F4, E4, E4) + ((E4, 1), (E4, 1))
                          + _q(E4, D4, D4, E4) + ((D4, 4), (G4, 4))
                          + (_q(E4, E4) + ((E4, 4),)) * 2 + _q(E4, G4) + ((C4, 3), (D4, 1), (E4, 8))
                          + _q(F4, F4) + ((F4, 3), (F4, 1)) + _q(F4, E4, E4) + ((E4, 1), (E4, 1))
                          + _q(G4, G4, F4, D4) + ((C4, 8),)), 110, 8),
    Tune("Happy Birthday", ((G4, 1), (G4, 1), (A4, 2), (G4, 2), (C5, 2), (B4, 4), (G4, 1), (G4, 1),
                            (A4, 2), (G4, 2), (D5, 2), (C5, 4), (G4, 1), (G4, 1),
                            (G5, 2), (E5, 2), (C5, 2), (B4, 2), (A4, 2), (F5, 1), (F5, 1),
                            (E5, 2), (C5, 2), (D5, 2), (C5, 6)), 100, 6),
    Tune("London Bridge", (((G4, 3), (A4, 1), (G4, 2), (F4, 2)) + _q(E4, F4) + ((G4, 4),) + _q(D4, E4) + ((F4, 4),)
                           + _q(E4, F4) + ((G4, 4),) + ((G4, 3), (A4, 1), (G4, 2), (F4, 2)) + _q(E4, F4)
                           + ((G4, 4), (D4, 4), (G4, 4), (E4, 2), (C4, 6))), 100, 8),
    Tune("Au clair de la lune", ((_q(C4, C4, C4, D4) + ((E4, 4), (D4, 4)) + _q(C4, E4, D4, D4) + ((C4, 8),)) * 2
                                 + _q(D4, D4, D4, D4) + ((A3, 4), (A3, 4)) + _q(D4, C4, B3, A3) + ((G3, 8),)
                                 + _q(C4, C4, C4, D4) + ((E4, 4), (D4, 4)) + _q(C4, E4, D4, D4) + ((C4, 8),)),
         90, 8),
    Tune("Чижик-пыжик", ((_q(E4, C4, E4, C4, F4, E4) + ((D4, 4),) + _q(G4, G4, G4, A4, B4, C5) + ((C5, 4),)) * 2),
         110, 8),
    Tune("Old MacDonald", (_q(C4, C4, C4, G3, A3, A3) + ((G3, 4),) + _q(E4, E4, D4, D4) + ((C4, 6), (G3, 2))
                           + _q(C4, C4, C4, G3, A3, A3) + ((G3, 4),) + _q(E4, E4, D4, D4) + ((C4, 8),)), 110, 8),
)


def by_name(name, tunes=BUILTIN):
    return next((t for t in tunes if t.name == name), None)


def in_range(tune):
    """Та же мелодия, перенесённая октавами в диапазон гитары (ближайший к исходному сдвиг)."""
    ps = [p for p, _ in tune.notes if p is not None]
    if not ps:
        return tune
    options = [k * 12 for k in range(-4, 5)
               if guitar.LOW <= min(ps) + k * 12 and max(ps) + k * 12 <= guitar.HIGH]
    if options:
        shift = min(options, key=abs)
        notes = tuple((None if p is None else p + shift, d) for p, d in tune.notes)
    else:                               # шире трёх октав - каждую ноту по отдельности
        def fold(p):
            while p < guitar.LOW:
                p += 12
            while p > guitar.HIGH:
                p -= 12
            return p
        notes = tuple((None if p is None else fold(p), d) for p, d in tune.notes)
    return Tune(tune.name, notes, tune.tempo, tune.meter, tune.builtin)


def phrases(tune):
    """Фразы по 4 такта в диапазоне гитары; нота - во фразе, где началась, хвост обрезан."""
    tune = in_range(tune)
    size = MEASURES_PER_PHRASE * tune.meter
    out, t = {}, 0
    for p, d in tune.notes:
        k = t // size
        if p is not None:
            start = t - k * size
            out.setdefault(k, []).append(Target(start, p, min(d, size - start)))
        t += d
    count = max(1, -(-t // size))
    return [Phrase(tune.name, k, tuple(out.get(k, ())), size, tune.tempo, tune.meter) for k in range(count)
            if out.get(k)]


def _tune_from(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"].strip():
        return None
    notes = raw.get("notes")
    if not isinstance(notes, list) or not notes:
        return None
    clean = []
    for item in notes:
        if not (isinstance(item, list) and len(item) == 2 and is_int(item[1]) and 1 <= item[1] <= 32):
            return None
        p = item[0]
        if p is not None and not (is_int(p) and 20 <= p <= 110):
            return None
        clean.append((p, item[1]))
    tempo = raw.get("tempo", 100)
    meter = raw.get("meter", 8)
    if not (is_int(tempo) and 40 <= tempo <= 200 and meter in (6, 8)):
        return None
    return Tune(raw["name"].strip(), tuple(clean), tempo, meter, builtin=False)


class Songbook:
    def __init__(self, path=None, own=()):
        self.path = path
        self.own = list(own)

    @classmethod
    def load(cls, path):
        raw = load_json(path)
        own = []
        if isinstance(raw, list):
            for item in raw:
                t = _tune_from(item)
                if t is not None and by_name(t.name) is None and by_name(t.name, own) is None:
                    own.append(t)
        return cls(path, own)

    def all(self):
        return list(BUILTIN) + self.own

    def by_name(self, name):
        return by_name(name, self.all())

    def add(self, tune):
        if by_name(tune.name) is not None:
            raise ValueError("так называется встроенная мелодия")
        self.own = [t for t in self.own if t.name != tune.name] + [tune]

    def remove(self, name):
        self.own = [t for t in self.own if t.name != name]

    def save(self):
        if self.path is None:
            return False
        return save_json(self.path, [{"name": t.name, "notes": [list(n) for n in t.notes], "tempo": t.tempo,
                                      "meter": t.meter} for t in self.own])
