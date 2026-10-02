"""Что живёт с человечком между мирами и запусками, кроме гитары и навыков: нужды, рюкзак,
что ему нравится и что надоело (выбор занятий), вкус к еде.

Файлы по зерну тела: life/<зерно>/state.json (нужды, рюкзак, выбор) и
skills/<зерно>/food.npz (вкус - сеть). Нет файла или битый - всё с нуля: нужды на нуле,
рюкзак полный (так спецификация 2.0 переносит сохранения 1.1.0).
"""
import json
import os

import paths
from acts.choice import Choice
from acts.eat import FoodPrefs
from acts.needs import Backpack, Needs
from skills.swim import Swimmer
from skills import store as skill_store


def state_path(body_seed):
    return paths.user_file("life", str(int(body_seed)), "state.json")


def load(body_seed, freq="sometimes", rng=None):
    """(нужды, рюкзак, выбор, вкус к еде, навык плавания) человечка."""
    raw = {}
    try:
        with open(state_path(body_seed), encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, ValueError):
        raw = {}
    needs = Needs().load(raw.get("needs"))
    pack = Backpack().load(raw.get("backpack"))
    choice = Choice(freq, rng).load(raw.get("choice"))
    prefs = skill_store.load(body_seed, "food", lambda: FoodPrefs(seed=int(body_seed) % (2 ** 31)))
    swimmer = skill_store.load(body_seed, "swim", lambda: Swimmer(seed=int(body_seed) % (2 ** 31)))
    return needs, pack, choice, prefs, swimmer


def save(life):
    data = {"needs": life.needs.to_dict(), "backpack": life.backpack.to_dict(), "choice": life.choice.to_dict()}
    path = state_path(life.body_seed)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        return False
    ok = skill_store.save(life.body_seed, "food", life.daily.prefs)
    ok = skill_store.save(life.body_seed, "swim", life.swim.swimmer) and ok
    for name, heads in life.toy_heads.items():
        try:
            heads.save(toy_path(life.body_seed, name))
        except OSError:
            ok = False
    for j in life.toy_journals.values():
        ok = j.save() and ok
    return ok


def toy_path(body_seed, name):
    return paths.user_file("toys", str(int(body_seed)), name + ".npz")


def load_toys(body_seed):
    """Головы игрушек человечка: {игрушка: головы}; нет или битые - с нуля (появятся при игре)."""
    out = {}
    from toys.ball import BallHeads
    for name, make in (("ball", lambda: BallHeads(seed=int(body_seed) % (2 ** 31))),):
        p = toy_path(body_seed, name)
        if os.path.exists(p):
            try:
                out[name] = make().load(p)
            except Exception:  # noqa: BLE001
                pass
    return out
