"""Навыки человечка на диске: skills/<зерно тела>/<навык>.npz в папке данных игры.

Новый мир с тем же человечком - те же навыки. Нет файла или он битый - навык с нуля (так и
задумано: у каждого человечка своё тело и свои навыки). Руки 1.1.0 лежали в
guitar/<зерно>/hands.npz - они читаются как навык reach, если нового файла ещё нет.
"""
import os

import paths

LEGACY = {"reach": ("guitar", "hands.npz")}


def path(body_seed, name):
    return paths.user_file("skills", str(int(body_seed)), name + ".npz")


def load(body_seed, name, make):
    """Навык name человечка body_seed; make() - новый (с нуля)."""
    candidates = [path(body_seed, name)]
    if name in LEGACY:
        folder, file = LEGACY[name]
        candidates.append(paths.user_file(folder, str(int(body_seed)), file))
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            return make().load(p)
        except Exception:  # noqa: BLE001 - битый файл: навык с нуля
            break
    return make()


def save(body_seed, name, skill):
    try:
        skill.save(path(body_seed, name))
        return True
    except OSError:
        return False
