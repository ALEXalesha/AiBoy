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
    return save_dog(life) and ok


def dog_paths(world_seed):
    """Собака живёт в своём мире: worlds/<зерно мира>/dog.json и её сети."""
    base = ("worlds", str(int(world_seed)))
    return (paths.user_file(*base, "dog.json"), paths.user_file(*base, "dog_motor.npz"),
            paths.user_file(*base, "dog_fetch.npz"))


def load_dog(world_seed):
    """(состояние, сеть движения, сеть «принести») собаки мира или None, если её тут не было.
    Битые сети - с нуля (состояние остаётся), битое состояние - как не было."""
    from animals.dog import motor as dog_motor
    from animals.dog.dog import Dog  # noqa: F401  (проверка, что модуль цел)
    jp, mp, fp = dog_paths(world_seed)
    try:
        with open(jp, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return None
    except (OSError, ValueError):
        return None
    motor = fetch = None
    try:
        motor = dog_motor.DogMotor(seed=int(world_seed) % (2 ** 31)).load(mp)
    except Exception:  # noqa: BLE001
        motor = None
    try:
        from acts.policy import ChoicePolicy
        from animals.dog.dog import FETCH_CHOICES
        fetch = ChoicePolicy(3, len(FETCH_CHOICES), hidden=8, seed=int(world_seed) % (2 ** 31) + 3, lr=8e-2, entropy=0.05)
        fetch.load(fp)
    except Exception:  # noqa: BLE001
        fetch = None
    return state, motor, fetch


def save_dog(life):
    if life.dog is not None:
        state, motor, fetch = life.dog.to_dict(), life.dog.motor, life.dog.fetch
    elif life.dog_saved:
        state, motor, fetch = life.dog_saved
    else:
        return True
    jp, mp, fp = dog_paths(life.world_seed)
    try:
        tmp = jp + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, jp)
        if motor is not None:
            motor.save(mp)
        if fetch is not None:
            fetch.save(fp)
    except OSError:
        return False
    return True


def toy_path(body_seed, name):
    return paths.user_file("toys", str(int(body_seed)), name + ".npz")


def load_toys(body_seed):
    """Головы игрушек человечка: {игрушка: головы}; нет или битые - с нуля (появятся при игре)."""
    out = {}
    from toys import registry
    from toys.ball import BallHeads
    from animals.dog.session import TameHeads
    makers = [("ball", lambda: BallHeads(seed=int(body_seed) % (2 ** 31))),
              ("dog", lambda: TameHeads(seed=int(body_seed) % (2 ** 31)))]
    for name in registry.INPLACE:
        heads_cls = registry.inplace(name)[0]
        makers.append((name, lambda c=heads_cls: c(seed=int(body_seed) % (2 ** 31))))
    for name, make in makers:
        p = toy_path(body_seed, name)
        if os.path.exists(p):
            try:
                out[name] = make().load(p)
            except Exception:  # noqa: BLE001
                pass
    return out
