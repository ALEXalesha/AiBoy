"""Награда попытки: R = wB·B + wA·A + wC·C.

- **B - приятно само по себе**: консонанс одновременных и соседних нот (таблицы
  интервалов), ровный ритм (удары на долях, ровные промежутки), новизна (непохожесть на
  его последние фразы), минус тишина и «каша» (слишком густо).
- **A - похоже на мелодию** (только когда учит мелодию): выравнивание сыгранных нот с
  нотами мелодии, как DTW - совпадение, лишняя нота, пропущенная нота; цена совпадения -
  разница высот и сдвиг во времени. Точный повтор - 1, пусто - 0.
- **C - оценка владельца** 👍 +1 / 👎 -1; без оценки - догадка модели вкуса (music/taste.py).
  Её вес самый большой.

Ноты - в шагах сетки (восьмых) от начала фразы.
"""
from dataclasses import dataclass

import numpy as np

WEIGHTS = {"A": 1.0, "B": 0.5, "C": 2.0}

# интервал между одновременными нотами (по модулю октавы) -> насколько приятно
HARMONIC = {0: 1.0, 7: 0.9, 5: 0.8, 4: 0.75, 3: 0.7, 9: 0.65, 8: 0.6, 10: 0.3, 2: 0.25, 6: 0.1, 11: 0.05, 1: 0.0}
# шаг мелодии между соседними нотами (полутонов) -> насколько естественно
MELODIC = {0: 0.6, 1: 0.9, 2: 1.0, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.2, 7: 0.7, 8: 0.5, 9: 0.5, 10: 0.35,
           11: 0.15, 12: 0.6}
HISTORY = 8                  # столько последних фраз помнит новизна
MIN_NOTES = 4
MUSH_PER_STEP = 0.9
MUSH_TOGETHER = 3


@dataclass(frozen=True)
class Played:
    onset: int
    pitch: int
    dur: int
    force: int


@dataclass(frozen=True)
class Pleasant:
    consonance: float
    rhythm: float
    novelty: float
    silence: float
    mush: float
    total: float


def _sorted(notes):
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def consonance(notes):
    """От 0 до 1: среднее по парам одновременных (гармония) и соседних по времени (мелодия)."""
    ns = _sorted(notes)
    vals = []
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            if b.onset >= a.onset + a.dur:
                break
            vals.append(HARMONIC[abs(b.pitch - a.pitch) % 12])
    onsets = sorted({n.onset for n in ns})
    tops = [max(n.pitch for n in ns if n.onset == t) for t in onsets]
    for a, b in zip(tops, tops[1:]):
        vals.append(MELODIC.get(abs(b - a), 0.2))
    return float(np.mean(vals)) if vals else 0.0


def rhythm(notes, meter):
    """От 0 до 1: доля ударов на долях (чётные восьмые) и ровность промежутков."""
    onsets = sorted({n.onset for n in notes})
    if len(onsets) < 2:
        return 0.0
    on_beat = float(np.mean([o % 2 == 0 for o in onsets]))
    gaps = np.diff(onsets)
    steady = float(np.clip(1.0 - gaps.std() / (gaps.mean() + 1e-9), 0.0, 1.0))
    return 0.5 * on_beat + 0.5 * steady


def _events(notes):
    return {(n.onset, n.pitch) for n in notes}


def novelty(notes, history):
    """От 0 до 1: насколько фраза не похожа на ближайшую из его последних (1 - доля общих
    пар «шаг, высота»)."""
    mine = _events(notes)
    if not history:
        return 1.0
    best = 0.0
    for other in history[-HISTORY:]:
        theirs = _events(other)
        union = mine | theirs
        if union:
            best = max(best, len(mine & theirs) / len(union))
    return 1.0 - best


def pleasant(notes, meter, history, steps=32):
    cons = consonance(notes)
    rhy = rhythm(notes, meter)
    nov = novelty(notes, history)
    silence = float(np.clip((MIN_NOTES - len(notes)) / MIN_NOTES, 0.0, 1.0))
    per_step = len(notes) / max(steps, 1)
    together = max([sum(1 for n in notes if n.onset == t) for t in {n.onset for n in notes}] or [0])
    mush = float(np.clip((per_step - MUSH_PER_STEP) / MUSH_PER_STEP, 0.0, 1.0))
    mush = max(mush, float(np.clip((together - MUSH_TOGETHER) / 3.0, 0.0, 1.0)))
    raw = 0.4 * cons + 0.3 * rhy + 0.3 * nov
    total = float(np.clip(2.0 * raw - 1.0 - 1.5 * silence - 1.5 * mush, -1.0, 1.0))
    return Pleasant(cons, rhy, nov, silence, mush, total)


def _match_cost(p, t):
    d = abs(p.pitch - t.pitch)
    pitch = 0.0 if d == 0 else (0.5 if d % 12 == 0 else min(1.0, 0.3 + d / 12.0))
    shift = min(1.0, abs(p.onset - t.onset) / 8.0)
    return pitch + 0.5 * shift


def closeness(played, target):
    """A от 0 до 1: выравнивание сыгранного с мелодией (как DTW, но нота с нотой - одна к
    одной; лишняя и пропущенная стоят 1)."""
    t = sorted(target, key=lambda n: n.onset)
    p = _sorted(played)
    if not t:
        return 0.0
    n, m = len(p), len(t)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + _match_cost(p[i - 1], t[j - 1]), D[i - 1, j] + 1.0, D[i, j - 1] + 1.0)
    return float(np.clip(1.0 - D[n, m] / m, 0.0, 1.0))


def total(a, b, c):
    """Итог: слагаемые, которых нет (A в свободной игре), не считаются."""
    r = WEIGHTS["B"] * b
    if a is not None:
        r += WEIGHTS["A"] * a
    if c is not None:
        r += WEIGHTS["C"] * c
    return float(r)
