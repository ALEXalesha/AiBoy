"""Обученные сети из папки models/ и паспорта к ним.

Если файла нет (только что склонированный репозиторий до train/*), берётся случайная сеть
с тем же устройством - игра работает, просто мир и речь хуже. trained показывает, что
загрузилось по-настоящему.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

import paths
from body import cppn as body_cppn
from brain.brain import Brain
from speech.net import SpeechNet
from voice.net import VoiceNet
from world import cppn as world_cppn


@dataclass
class Models:
    world_genome: np.ndarray
    body_genome: np.ndarray
    speech: SpeechNet
    voice: VoiceNet
    trained: dict = field(default_factory=dict)


def model_path(name):
    return paths.resource("models", name)


def passport(name):
    try:
        with open(model_path(name + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _genome(name, size, fallback, trained):
    try:
        with np.load(model_path(name + ".npz")) as data:
            g = data["genome"].astype(float)
        if g.shape == (size,):
            trained[name] = True
            return g
    except (OSError, KeyError, ValueError):
        pass
    trained[name] = False
    return fallback


_cache = None


def load():
    global _cache
    if _cache is not None:
        return _cache
    trained = {}
    world = _genome("world", world_cppn.GENOME_SIZE, world_cppn.random_genome(np.random.default_rng(3)), trained)
    body = _genome("body", body_cppn.GENOME_SIZE, body_cppn.random_genome(np.random.default_rng(2)), trained)
    speech = SpeechNet(seed=0)
    try:
        speech.load(model_path("speech.npz"))
        trained["speech"] = True
    except (OSError, KeyError, ValueError):
        trained["speech"] = False
    voice = VoiceNet(seed=0)
    try:
        voice.load(model_path("voice.npz"))
        trained["voice"] = True
    except (OSError, KeyError, ValueError):
        trained["voice"] = False
    trained["brain"] = os.path.exists(model_path("brain.npz"))
    _cache = Models(world, body, speech, voice, trained)
    return _cache


def starter_brain():
    """Стартовый мозг из models/brain.npz (немного обучен при сборке), иначе новый."""
    brain = Brain(seed=0)
    try:
        brain.load(model_path("brain.npz"))
    except Exception:  # noqa: BLE001 - нет файла или он чужой - начинаем с нуля
        brain = Brain(seed=0)
    return brain


def user_brain(path):
    """Мозг игрока из папки данных; нет или битый - стартовый."""
    brain = Brain(seed=0)
    try:
        brain.load(path)
        return brain, True
    except Exception:  # noqa: BLE001 - битый файл не должен ронять игру
        return starter_brain(), False
