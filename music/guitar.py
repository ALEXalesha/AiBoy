"""Гитара: шесть струн в обычном строе E2 A2 D3 G3 B3 E4, лады 0-12, три силы удара.

Нота - (струна, лад, начало в секундах, сила 0..2, длительность до глушения, приглушена ли).
Высота - номер MIDI: открытая струна плюс лад.
"""
from dataclasses import dataclass

OPEN = (40, 45, 50, 55, 59, 64)          # E2 A2 D3 G3 B3 E4, струна 0 - самая низкая
STRINGS = len(OPEN)
FRETS = 13                                # 0..12
FORCES = 3
LOW, HIGH = OPEN[0], OPEN[-1] + FRETS - 1   # E2 .. E5
NAMES = ("до", "до#", "ре", "ре#", "ми", "фа", "фа#", "соль", "соль#", "ля", "ля#", "си")


def midi(string, fret):
    return OPEN[string] + fret


def freq(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def positions(pitch):
    """Все (струна, лад), где звучит эта высота."""
    return [(s, pitch - o) for s, o in enumerate(OPEN) if 0 <= pitch - o < FRETS]


def name(pitch):
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass(frozen=True)
class Note:
    string: int
    fret: int
    onset: float          # с от начала фразы
    force: int            # 0 тихо, 1 средне, 2 сильно
    duration: float       # с до глушения
    muted: bool = False   # приглушённая (едва задел струну)

    @property
    def pitch(self):
        return midi(self.string, self.fret)
