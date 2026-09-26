"""Буквы речи: русский алфавит, пробел и пять знаков. 0 - конец фразы, 1 - начало."""
LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
PUNCT = " .,!?-"
CHARS = LETTERS + PUNCT
END = 0
START = 1
SIZE = 2 + len(CHARS)
MAX_LEN = 40
_INDEX = {c: i + 2 for i, c in enumerate(CHARS)}


def encode(text):
    return [_INDEX[c] for c in text.lower() if c in _INDEX]


def decode(ids):
    out = []
    for i in ids:
        if i == END:
            break
        if i >= 2:
            out.append(CHARS[i - 2])
    return "".join(out)
