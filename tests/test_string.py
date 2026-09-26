"""Звук струны: Карплус-Стронг с корпусом. Законы: своя высота (не хуже 5 центов), затухание
без подъёмов, сила - громкость, приглушённая - тише и короче, фраза без перегруза."""
import time

import numpy as np
import pytest

from music import guitar, string
from music.guitar import Note

SR = string.SR


def cents_off(samples, expected_hz):
    """Отклонение основного тона от ожидаемого, центы: пик спектра в полутоне от ожидаемой
    частоты, с дополнением нулями и параболой по логарифму амплитуды."""
    x = samples[int(0.05 * SR):int(1.05 * SR)].astype(np.float64)
    x = x * np.hanning(len(x))
    n = len(x) * 8
    spec = np.abs(np.fft.rfft(x, n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    band = (freqs > expected_hz * 2 ** (-1 / 12)) & (freqs < expected_hz * 2 ** (1 / 12))
    idx = np.flatnonzero(band)
    k = idx[np.argmax(spec[idx])]
    a, b, c = np.log(spec[k - 1:k + 2] + 1e-12)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    f = (k + shift) * SR / n
    return 1200 * np.log2(f / expected_hz), spec[k] / spec.max()


@pytest.mark.parametrize("s", range(6))
@pytest.mark.parametrize("fret", [0, 5, 12])
def test_every_string_and_fret_sounds_at_its_own_pitch(s, fret):
    tone = string.pluck(guitar.freq(guitar.midi(s, fret)), force=1, seed=s * 13 + fret)
    off, share = cents_off(tone, guitar.freq(guitar.midi(s, fret)))
    assert abs(off) <= 5.0, off
    assert share > 0.1                        # основной тон слышен, а не только обертоны


def envelope(x, win=0.05):
    n = int(win * SR)
    k = len(x) // n
    return np.sqrt((x[:k * n].reshape(k, n) ** 2).mean(axis=1))


@pytest.mark.parametrize("s", [0, 3, 5])
def test_the_sound_decays_without_swelling(s):
    env = envelope(string.pluck(guitar.freq(guitar.midi(s, 3)), force=2))
    after = env[env.argmax():]
    assert (np.diff(after) <= 1e-4).all()
    assert after[-1] < 0.3 * after[0]


def test_stronger_pluck_is_louder():
    f = guitar.freq(guitar.midi(2, 2))
    loud = [np.sqrt((string.pluck(f, force=k)[:int(0.2 * SR)] ** 2).mean()) for k in range(3)]
    assert loud[0] < loud[1] < loud[2]


def test_muted_note_is_quieter_and_shorter():
    f = guitar.freq(guitar.midi(1, 0))
    open_, muted = string.pluck(f, 1), string.pluck(f, 1, muted=True)
    assert np.abs(muted).max() < 0.6 * np.abs(open_).max()
    tail = slice(int(0.5 * SR), int(1.0 * SR))
    assert np.abs(muted[tail]).max() < 0.05 * np.abs(open_[tail]).max()


def test_phrase_is_mixed_without_clipping():
    rng = np.random.default_rng(0)
    notes = [Note(int(rng.integers(6)), int(rng.integers(13)), 0.12 * i, 2, 0.8) for i in range(30)]
    notes += [Note(s, 0, 1.0, 2, 1.0) for s in range(6)]           # полный аккорд разом
    bank = string.StringBank()
    audio = bank.render(notes)
    assert audio.dtype == np.float32 and np.isfinite(audio).all()
    assert np.abs(audio).max() <= 0.95
    assert len(audio) >= int((0.12 * 29 + 0.8) * SR)


def test_note_stops_softly_after_its_duration():
    bank = string.StringBank()
    audio = bank.render([Note(0, 0, 0.0, 2, 0.3)], total=1.0)
    stop = int((0.3 + string.RELEASE) * SR)
    assert np.abs(audio[stop + 10:]).max() < 1e-3
    edge = audio[int(0.3 * SR):stop]
    assert np.abs(np.diff(edge)).max() < 0.05                 # без щелчка


def test_bank_caches_and_warms_quickly():
    bank = string.StringBank()
    a = bank.sample(2, 5, 1)
    assert bank.sample(2, 5, 1) is a
    started = time.perf_counter()
    bank.warm()
    assert time.perf_counter() - started < 10.0
    assert len(bank.cache) == 6 * 13 * 3 * 2


def test_pitch_math():
    assert guitar.midi(0, 0) == 40 and guitar.midi(5, 12) == 76
    assert guitar.freq(69) == pytest.approx(440.0)
    assert set(guitar.positions(52)) == {(0, 12), (1, 7), (2, 2)}
    assert guitar.positions(30) == []
