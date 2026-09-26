"""Синтезатор голоса на numpy: 7 чисел от 0 до 1 -> короткий звук.

- pitch - высота: ступень пентатоники (до, ре, ми, соль, ля) от соль малой октавы;
- glide - куда ведёт мелодия: 0 - вниз, 0.5 - на месте, 1 - вверх (до трёх ступеней на ноту);
- notes - одна, две или три ноты;
- length - длина, от 0.16 до 0.6 с;
- vowel - гласная: у, о, а, э, и (форманты - какие гармоники громче);
- vibrato - дрожание высоты; bright - яркость (сколько высоких гармоник).

Звук мягкий: ноты только из пентатоники (любые две звучат вместе без фальши), тихое
нарастание и затухание, между нотами - скольжение, а не ступенька. Громкость не выше 0.55.
"""
import io
import wave

import numpy as np

RATE = 22050
PARAMS = ("pitch", "glide", "notes", "length", "vowel", "vibrato", "bright")
PENTATONIC = (0, 2, 4, 7, 9)
BASE = 196.0                     # соль малой октавы
DEGREES = 12                     # ступеней по высоте - две с лишним октавы
MIN_LEN, MAX_LEN = 0.16, 0.6
PEAK = 0.55
# форманты гласных у, о, а, э, и (Гц)
VOWELS = ((320, 800), (500, 900), (750, 1250), (550, 1800), (300, 2300))


def _p(p, name):
    return float(np.clip(p[PARAMS.index(name)], 0.0, 1.0))


def length_samples(p):
    return round((MIN_LEN + (MAX_LEN - MIN_LEN) * _p(p, "length")) * RATE)


def degree_freq(d):
    octave, step = divmod(int(d), len(PENTATONIC))
    return BASE * 2.0 ** ((12 * octave + PENTATONIC[step]) / 12.0)


def note_freqs(p):
    n = 1 + min(2, int(_p(p, "notes") * 3))
    start = round(_p(p, "pitch") * (DEGREES - 1))
    step = round((_p(p, "glide") - 0.5) * 6)          # -3..3 ступени на ноту
    return [degree_freq(int(np.clip(start + i * step, 0, DEGREES + 2))) for i in range(n)]


def _formants(p):
    v = _p(p, "vowel") * (len(VOWELS) - 1)
    i = min(int(v), len(VOWELS) - 2)
    t = v - i
    return [a + (b - a) * t for a, b in zip(VOWELS[i], VOWELS[i + 1])]


def synth(p):
    p = np.asarray(p, float)
    n = length_samples(p)
    freqs = note_freqs(p)
    t = np.arange(n) / RATE
    # частота по времени: ноты подряд, между ними скольжение 30 мс
    per = n / len(freqs)
    f = np.empty(n)
    for i, fr in enumerate(freqs):
        f[int(i * per):int((i + 1) * per)] = fr
    slide = max(1, int(0.03 * RATE))
    kernel = np.ones(slide) / slide
    f = np.convolve(np.pad(f, (slide, slide), mode="edge"), kernel, mode="same")[slide:slide + n]
    # падающая мелодия ещё и чуть съезжает вниз внутри нот, растущая - вверх
    f *= 2.0 ** ((_p(p, "glide") - 0.5) * 0.25 * t / max(t[-1], 1e-3))
    vib = 1.0 + 0.012 * _p(p, "vibrato") * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.08, 0, 1)
    phase = 2 * np.pi * np.cumsum(f * vib) / RATE
    f1, f2 = _formants(p)
    bright = 0.35 + 0.5 * _p(p, "bright")
    wave_ = np.zeros(n)
    f0 = float(np.mean(freqs))
    for k in range(1, 11):
        fk = k * f0
        gain = bright ** (k - 1) * (0.25 + np.exp(-((fk - f1) / 180.0) ** 2)
                                    + 0.6 * np.exp(-((fk - f2) / 260.0) ** 2))
        wave_ += gain * np.sin(k * phase)
    # огибающая: мягко каждая нота и весь звук
    env = np.ones(n)
    note = np.arange(n) % max(1, int(per))
    env *= np.clip(note / (0.012 * RATE), 0, 1) * 0.35 + 0.65
    env *= np.exp(-1.2 * t / max(t[-1], 1e-3))
    a, r = max(1, int(0.012 * RATE)), max(1, int(0.05 * RATE))
    env[:a] *= np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r) ** 2
    out = wave_ * env
    peak = np.abs(out).max()
    if peak > 0:
        out = out * (PEAK / peak)
    return out.astype(np.float32)


def to_wav(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
