"""Учитель голоса - только для обучения (train/voice.py), в игру не попадает.

По сводке состояния даёт 7 параметров синтезатора (voice/synth.py). Тема - самая важная
из того, что происходит: упал, летит или хочет прыгнуть, рядом сущность, любопытно,
скучно, ночь. Немного шума - чтобы звуки одной темы не были одинаковыми.
"""
import numpy as np

from brain.state import IDX
from voice.synth import PARAMS

#            pitch glide notes length vowel vibrato bright
FALL = (0.15, 0.1, 0.5, 0.75, 0.25, 0.35, 0.3)       # «о-ой» вниз
JUMP = (0.85, 0.95, 0.1, 0.05, 0.95, 0.0, 0.7)       # короткое высокое «и!»
NEAR = (0.6, 0.75, 0.9, 0.5, 0.5, 0.2, 0.6)          # «а-а-а» вверх, три ноты
CURIOUS = (0.5, 0.9, 0.5, 0.35, 0.05, 0.25, 0.45)    # «у-у?» вверх
BORED = (0.3, 0.35, 0.2, 0.55, 0.3, 0.1, 0.2)        # тихое «мм»
NIGHT = (0.4, 0.45, 0.5, 0.65, 0.2, 0.55, 0.25)      # мягко, с дрожью
IDLE = (0.5, 0.6, 0.5, 0.3, 0.5, 0.2, 0.5)


def topic(state):
    s = lambda name: float(state[IDX[name]])  # noqa: E731
    if s("fell") > 0.5:
        return FALL
    if s("airborne") > 0.5 or s("want_jump") > 0.5:
        return JUMP
    if s("entity") > 0.5 and s("entity_dist") < 0.25:
        return NEAR
    if s("curiosity") > 0.6:
        return CURIOUS
    if s("curiosity") < 0.25:
        return BORED
    if s("night") > 0.5:
        return NIGHT
    return IDLE


def target(state, rng, noise=0.05):
    base = np.array(topic(state), float)
    assert len(base) == len(PARAMS)
    return np.clip(base + rng.normal(0.0, noise, len(base)), 0.0, 1.0)
