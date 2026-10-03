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


def scratch(force=1.0, size=0.05, seed=7):
    """Фломастер по бумаге: короткий светлый шорох."""
    n = int(0.12 * SR)
    t = np.arange(n) / n
    x = _lowpass(_noise(n, seed), 0.7) * np.sin(np.pi * t)
    return (0.15 * force * x).astype(np.float32)


def chain(force=1.0, size=0.1, seed=8):
    """Цепь велосипеда: тихий тик."""
    n = int(0.04 * SR)
    x = _tone(2400, n, 0.004) + 0.3 * _noise(n, seed) * _env(n, attack=0.0002, decay=0.002)
    return (0.2 * force * x).astype(np.float32)


def bark(force=1.0, size=0.6, seed=9):
    """Лай: короткий тон с падающей высотой и хрипотцой; больше собака - ниже."""
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f0 = 900 - 650 * min(1.0, max(0.0, (size - 0.4) / 0.5))
    f = f0 * (1.0 - 0.35 * t / t[-1])
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.3 * _lowpass(_noise(n, seed), 0.4)) * _env(n, attack=0.004, decay=0.05)
    return (0.4 * force * x).astype(np.float32)


def shake(force=1.0, size=0.6, seed=10):
    """Отряхивается: брызги, быстро дрожащий шум."""
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    x = _lowpass(_noise(n, seed), 0.3) * (0.5 + 0.5 * np.sin(2 * np.pi * 18 * t)) * _env(n, attack=0.02, decay=0.25)
    return (0.3 * force * x).astype(np.float32)


def crunch(force=1.0, size=0.05, seed=11):
    """Хрустит лакомством."""
    n = int(0.15 * SR)
    x = _lowpass(_noise(n, seed), 0.6) * _env(n, attack=0.001, decay=0.02)
    return (0.3 * force * x).astype(np.float32)


def fanfare(force=1.0, size=0.1, seed=12):
    """Вершина: три восходящих тона - ура."""
    parts = []
    for f in (523.0, 659.0, 784.0):
        n = int(0.14 * SR)
        parts.append(_tone(f, n, 0.12) + 0.3 * _tone(2 * f, n, 0.06))
    return (0.35 * force * np.concatenate(parts)).astype(np.float32)


def flag(force=1.0, size=0.1, seed=13):
    """Флаг воткнут: глухой стук и хлопок ткани."""
    n = int(0.25 * SR)
    x = _tone(140.0, n, 0.05) + 0.4 * _lowpass(_noise(n, seed), 0.3) * _env(n, attack=0.03, decay=0.08)
    return (0.4 * force * x).astype(np.float32)


def clink(force=1.0, size=0.05, seed=14):
    """Крюк зацепился: металлический звон."""
    n = int(0.2 * SR)
    x = _tone(2200.0, n, 0.08) + 0.5 * _tone(3300.0, n, 0.05)
    return (0.25 * force * x).astype(np.float32)


def strike(force=1.0, size=0.05, seed=15):
    """Удар ледоруба по камню."""
    n = int(0.1 * SR)
    x = _tone(900.0, n, 0.02) + 0.6 * _noise(n, seed) * _env(n, attack=0.0005, decay=0.01)
    return (0.35 * force * x).astype(np.float32)


def scrape(force=1.0, size=0.05, seed=16):
    """Подтянулся: шорох по камню."""
    n = int(0.15 * SR)
    t = np.arange(n) / n
    x = _lowpass(_noise(n, seed), 0.4) * np.sin(np.pi * t)
    return (0.12 * force * x).astype(np.float32)


def bubbles(force=1.0, size=0.05, seed=17):
    """Пузыри из-под воды: короткие высокие бульки."""
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for k in range(6):
        s0 = int(rng.uniform(0, 0.32) * SR)
        m = min(n - s0, int(0.05 * SR))
        f = rng.uniform(500, 1200)
        x[s0:s0 + m] += np.sin(2 * np.pi * f * t[:m] * (1 + 2 * t[:m])) * np.exp(-t[:m] * 60)
    return (0.25 * force * x).astype(np.float32)


def flap(force=1.0, size=0.3, seed=18):
    """Взмахи крыльев: несколько глухих хлопков, у крупных реже и ниже."""
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    rate = 14.0 if size < 0.4 else 7.0
    x = _lowpass(_noise(n, seed), 0.2) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)) ** 4 * _env(n, attack=0.01, decay=0.2)
    return (0.4 * force * x).astype(np.float32)


def chirp(force=1.0, size=0.3, seed=19):
    """Щебет: два коротких свиста вверх, мелкие - выше."""
    parts = []
    base = 2600.0 if size < 0.4 else 1500.0
    for k in range(2):
        n = int(0.07 * SR)
        t = np.arange(n) / SR
        f = base * (1.0 + 0.6 * t / t[-1]) * (1.0 + 0.1 * k)
        parts.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, attack=0.005, decay=0.03))
        parts.append(np.zeros(int(0.04 * SR)))
    return (0.2 * force * np.concatenate(parts)).astype(np.float32)


SOUNDS = {"flap": flap, "chirp": chirp, "bubbles": bubbles, "fanfare": fanfare, "flag": flag, "clink": clink, "strike": strike, "scrape": scrape, "bark": bark, "shake": shake, "treat": crunch, "scratch": scratch, "chain": chain, "kick": thump, "throw": whoosh, "bounce": bounce, "catch": slap, "land": thump, "clack": clack,
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
