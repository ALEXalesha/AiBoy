"""Звук гитарной струны: физическая модель щипка (Карплус-Стронг) и корпус.

Щипок - короткий шум в линии задержки длиной в один период. На каждом обороте волна
проходит петлю потерь: усреднение двух соседних отсчётов (струна глушит высокие частоты
сильнее низких) и дробная задержка (без неё высота округлялась бы до целого числа
отсчётов - на верхних ладах это десятки центов). Длина петли подобрана так, чтобы
период был ровно SR / f. Потом звук проходит через корпус - короткий отклик из
нескольких затухающих резонансов (воздух около 100 Гц, дека около 200 и 400 Гц).

Петля считается кусками длиной в задержку: каждый отсчёт зависит только от отсчётов не
ближе задержки назад, поэтому целый кусок - одно действие numpy, а не цикл по отсчётам.

Сила удара - громкость и яркость (слабый щипок - сглаженный шум, темнее). Приглушённая
нота - быстрое затухание и тише.
"""
import threading

import numpy as np

from music import guitar

SR = 44100
CACHE_SECONDS = 2.5
RELEASE = 0.04               # с - мягкое глушение после длительности ноты
FORCE_GAIN = (0.35, 0.65, 1.0)
FORCE_SMOOTH = (6, 3, 1)     # ширина сглаживания шума щипка: слабый - темнее
MIX_GAIN = 0.32
LIMIT = 0.95


def _t60(f):
    """Сколько секунд звучит струна до -60 дБ: низкие дольше, высокие короче."""
    return float(np.clip(4.5 * (82.4 / f) ** 0.55, 1.0, 5.0))


def _body():
    t = np.arange(int(0.06 * SR)) / SR
    ir = np.zeros_like(t)
    ir[0] = 1.0
    for f, g, tau in ((98.0, 0.035, 0.018), (205.0, 0.03, 0.012), (410.0, 0.02, 0.008)):
        ir += g * np.exp(-t / tau) * np.sin(2 * np.pi * f * t)
    return ir


BODY = _body()


def pluck(f, force=1, seconds=CACHE_SECONDS, muted=False, seed=0):
    """Одна нота частоты f: float32, пик не больше 1."""
    rng = np.random.default_rng(seed)
    period = SR / f
    M = int(np.floor(period - 0.5))
    frac = period - 0.5 - M                       # дробная задержка сверх целых отсчётов и 0.5 усреднения
    c = np.convolve([0.5, 0.5], [1.0 - frac, frac])  # три отвода петли
    t60 = 0.15 if muted else _t60(f)
    rho = 10.0 ** (-3.0 / (t60 * f))
    # потери на основной частоте у фильтра петли уже есть - поправка, чтобы затухание было t60
    w = 2 * np.pi * f / SR
    loss = abs(c[0] + c[1] * np.exp(-1j * w) + c[2] * np.exp(-2j * w))
    rho = min(0.9999, rho / max(loss, 1e-6))
    n = int(seconds * SR)
    y = np.zeros(n + M + 2)
    exc = rng.uniform(-1.0, 1.0, M + 2)
    k = FORCE_SMOOTH[force] + (4 if muted else 0)
    if k > 1:
        exc = np.convolve(exc, np.ones(k) / k, mode="same")
    shift = max(1, int(0.13 * (M + 2)))             # щипок у подставки: гребёнка по месту
    exc = exc - np.roll(exc, shift) * 0.3
    exc -= exc.mean()
    exc *= FORCE_GAIN[force] / (np.abs(exc).max() + 1e-12)
    if muted:
        exc *= 0.4
    y[:M + 2] = exc
    pos = M + 2
    while pos < len(y):
        end = min(pos + M, len(y))
        L = end - pos
        y[pos:end] = rho * (c[0] * y[pos - M:pos - M + L] + c[1] * y[pos - M - 1:pos - M - 1 + L]
                            + c[2] * y[pos - M - 2:pos - M - 2 + L])
        pos = end
    size = 1 << int(np.ceil(np.log2(n + len(BODY))))     # свёртка с корпусом через БПФ
    out = np.fft.irfft(np.fft.rfft(y[:n], size) * np.fft.rfft(BODY, size), size)[:n]
    fade = int(0.003 * SR)                          # первые 3 мс - без щелчка
    out[:fade] *= np.linspace(0.0, 1.0, fade)
    return out.astype(np.float32)


class StringBank:
    """Кэш нот: (струна, лад, сила, приглушена) -> звук на CACHE_SECONDS. Потокобезопасен:
    греется в фоне, читается из окна."""

    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.warming = False

    def sample(self, string, fret, force, muted=False):
        key = (string, fret, force, bool(muted))
        with self.lock:
            got = self.cache.get(key)
        if got is None:
            got = pluck(guitar.freq(guitar.midi(string, fret)), force, muted=muted,
                        seed=string * 100 + fret * 7 + force)
            with self.lock:
                got = self.cache.setdefault(key, got)
        return got

    def warm(self):
        for s in range(guitar.STRINGS):
            for fr in range(guitar.FRETS):
                for fo in range(guitar.FORCES):
                    for muted in (False, True):
                        self.sample(s, fr, fo, muted)

    def note_audio(self, note):
        src = self.sample(note.string, note.fret, note.force, note.muted)
        keep = min(len(src), int((note.duration + RELEASE) * SR))
        audio = src[:keep].copy()
        start = min(keep, int(note.duration * SR))
        tail = keep - start
        if tail > 0:
            audio[start:] *= np.cos(np.linspace(0.0, np.pi / 2, tail)) ** 2
        return audio

    def render(self, notes, total=None):
        """Фраза в звук: ноты смешаны по началам, общий уровень MIX_GAIN, пик не выше LIMIT."""
        end = max([n.onset + n.duration + RELEASE for n in notes] + [total or 0.0])
        out = np.zeros(int(end * SR) + 1, np.float32)
        for n in notes:
            a = self.note_audio(n)
            i = int(round(n.onset * SR))
            out[i:i + len(a)] += a[:len(out) - i]
        out *= MIX_GAIN
        peak = float(np.abs(out).max()) if len(out) else 0.0
        if peak > LIMIT:
            out *= np.float32(LIMIT * 0.999 / peak)
        return out


_shared = None


def shared_bank():
    """Один кэш на процесс: окон может быть несколько (тесты), струны у всех одни."""
    global _shared
    if _shared is None:
        _shared = StringBank()
    return _shared
