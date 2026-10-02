"""Звуки событий: синтез на numpy, как струна 1.1.0, - импульс, фильтрованный шум, тон
с огибающей; высота - по размеру и силе. Без файлов-сэмплов.

Каждый звук - функция (сила 0..1, размер м) -> отсчёты float32 44.1 кГц; SOUNDS - по имени
события. Окно играет их в канале «toys» микшера, только на x1-x2 (на x4-x16 - пропускает,
гитара держит свой x1).
"""
import numpy as np

SR = 44100


def _env(n, attack=0.002, decay=0.08):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return (a * np.exp(-t / decay)).astype(np.float32)


def _noise(n, seed):
    return np.random.default_rng(seed).normal(0.0, 1.0, n).astype(np.float32)


def _lowpass(x, k):
    """Однополюсный фильтр: k 0..1 - чем меньше, тем глуше."""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += k * (v - acc)
        y[i] = acc
    return y


def _tone(freq, n, decay):
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)).astype(np.float32)


def thump(force=1.0, size=0.2, seed=0):
    """Удар ногой по мячу, падение предмета: глухой низкий удар."""
    n = int(0.18 * SR)
    f = 60 + 90 / max(size, 0.05) ** 0.5
    x = 0.7 * _tone(f, n, 0.05) + 0.3 * _lowpass(_noise(n, seed), 0.08) * _env(n, decay=0.03)
    return (0.6 * force * x).astype(np.float32)


def bounce(force=1.0, size=0.11, seed=1):
    """Отскок мяча: короткий упругий «бум», выше у маленького мяча."""
    n = int(0.12 * SR)
    f = 180 + 120 / max(size, 0.05)
    x = _tone(f, n, 0.03) * 0.8 + 0.2 * _lowpass(_noise(n, seed), 0.3) * _env(n, decay=0.01)
    return (0.5 * min(1.0, force) * x).astype(np.float32)


def slap(force=1.0, size=0.1, seed=2):
    """Поймал: хлопок ладонью - светлый шум."""
    n = int(0.08 * SR)
    x = _lowpass(_noise(n, seed), 0.6) * _env(n, attack=0.0005, decay=0.012)
    return (0.5 * force * x).astype(np.float32)


def clack(force=1.0, size=0.15, seed=3):
    """Кубик о кубик: деревянный щелчок."""
    n = int(0.1 * SR)
    f = 600 + 900 * (0.2 / max(size, 0.05))
    x = _tone(f, n, 0.015) * 0.7 + 0.3 * _lowpass(_noise(n, seed), 0.5) * _env(n, decay=0.008)
    return (0.5 * force * x).astype(np.float32)


def splash(force=1.0, size=0.2, seed=4):
    """Всплеск: шум, приглушённый, с хвостом."""
    n = int(0.5 * SR)
    x = _lowpass(_noise(n, seed), 0.15) * _env(n, attack=0.005, decay=0.15)
    return (0.5 * force * x).astype(np.float32)


def whoosh(force=1.0, size=0.3, seed=5):
    """Взмах, бросок: шум, нарастает и спадает."""
    n = int(0.3 * SR)
    t = np.arange(n) / n
    x = _lowpass(_noise(n, seed), 0.25) * np.sin(np.pi * t) ** 2
    return (0.35 * force * x).astype(np.float32)


def pop(force=1.0, size=0.05, seed=6):
    """Лопнул пузырь: щелчок, выше у маленького."""
    n = int(0.05 * SR)
    f = 900 + 3000 * (0.05 / max(size, 0.01))
    x = _tone(min(f, 6000), n, 0.006) + 0.5 * _noise(n, seed) * _env(n, attack=0.0002, decay=0.003)
    return (0.4 * force * x).astype(np.float32)


SOUNDS = {"kick": thump, "throw": whoosh, "bounce": bounce, "catch": slap, "land": thump, "clack": clack,
          "fall": thump, "splash": splash, "whoosh": whoosh, "pop": pop}


_cache = {}


def make(event, force=1.0, size=0.15):
    """Звук события; одинаковые (с точностью до десятой силы и сантиметра размера) - из кэша:
    фильтр шума - цикл на Python, а мяч скачет часто."""
    f = SOUNDS.get(event)
    if f is None:
        return None
    key = (event, round(float(np.clip(force, 0.05, 1.0)), 1), round(float(size), 2))
    if key not in _cache:
        _cache[key] = f(key[1], key[2])
    return _cache[key]
