"""Музыкант живёт с человечком: веса сети и журнал попыток - в папке guitar/<зерно тела>
в данных игры. Новый мир с тем же человечком - тот же музыкант, та же гитара за спиной.
Битые файлы - новый музыкант со случайными весами и пустой журнал (так и задумано: у
каждого человечка своя музыка с нуля)."""
import os

import numpy as np

import paths
from skills import store as skill_store
from skills.reach import Reacher
from music.journal import Journal
from music.musician import Musician
from music.session import Guitarist


def folder(body_seed):
    path = paths.user_file("guitar", str(int(body_seed)), "x")
    return os.path.dirname(path)


def load(body_seed, book, mode="musician"):
    base = folder(body_seed)
    journal = Journal.load(os.path.join(base, "journal.json"))
    musician = Musician(seed=int(body_seed) % (2 ** 31))
    try:
        musician.load(os.path.join(base, "musician.npz"))
    except Exception:  # noqa: BLE001 - нет файла или битый: музыкант с нуля
        musician = Musician(seed=int(body_seed) % (2 ** 31))
    # руки - навык тела (skills/<зерно>/reach.npz; руки 1.1.0 из hands.npz читаются тоже)
    reacher = skill_store.load(body_seed, "reach", lambda: Reacher(seed=int(body_seed) % (2 ** 31)))
    g = Guitarist(musician, journal, book, mode=mode, reacher=reacher,
                  rng=np.random.default_rng([int(body_seed) & 0xFFFFFFFF, 11]))
    g.body_seed = int(body_seed)
    return g


def save(g):
    base = folder(g.body_seed)
    ok = g.journal.save()
    try:
        g.musician.save(os.path.join(base, "musician.npz"))
    except OSError:
        return False
    if g.reacher is not None:
        ok = skill_store.save(g.body_seed, "reach", g.reacher) and ok
    return ok


def has_guitar(g):
    return bool(g.journal.meta.get("has_guitar", False))


def set_has_guitar(g, value):
    g.journal.meta["has_guitar"] = bool(value)
