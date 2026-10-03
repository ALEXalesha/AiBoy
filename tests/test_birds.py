"""Птицы (спецификация 2.0, раздел 9): полёт учится, взлетают раньше, чем подойдут, никто не
ловит птицу, стайки - по настроению, звуки, собака гоняет севших."""
import math
import os
import tempfile

import numpy as np
import pytest

from animals.birds import flight as fl
from animals.birds import sky as sk
from animals.birds.flock import CATCH
from animals.birds.look import BirdLook, look
from animals.dog.practice import Flat
from sound import sfx


def test_look_and_kind_come_from_the_bird_network_by_world_seed():
    many = [look(s) for s in range(60)]
    assert look(3) == look(3) and look(3) != look(4)
    gliders = sum(b.glider for b in many)
    assert 10 < gliders < 50                                   # и парители, и мелкие
    assert len({b.color for b in many}) == 60


@pytest.mark.parametrize("glider,seed", [(False, 0), (False, 2), (True, 0)])
def test_flight_holds_height_better_after_training(glider, seed):
    """Сеть полёта без шума на 8 одинаковых полётах до и после 200 (замер, высота 3.5 м, зёрна 0-4:
    промах новичка 0.7-4 м; через 300 полётов у 7 из 10 - 0.12-0.48 м, у трёх - 1.6-2.4 м)."""
    lk = BirdLook(0.5, (0.5, 0.5, 0.5), 0.5, glider)
    f = fl.Flyer(seed)

    def judge():
        f.skill.explore = False
        r = np.mean([fl.practice(f, lk, np.random.default_rng(100 + i), learn=False) for i in range(8)])
        f.skill.explore = True
        return float(r)
    before = judge()
    rng = np.random.default_rng(seed)
    for _ in range(200):
        fl.practice(f, lk, rng)
    after = judge()
    assert after < 0.5 * before and after < 1.0, (before, after)


def approach_runs(seed, reps, speed=3.0):
    w = Flat()
    sky = sk.Sky(w, 5, "often", np.random.default_rng(seed))
    sky.takeoff.skill.explore = False
    d0 = sky.takeoff.decide(speed, False)[0]
    sky.takeoff.skill.explore = True
    closest, scares = math.inf, []
    for _ in range(reps):
        sky.flocks = []
        f = sky.spawn(100.0, 0.5)
        px, t = None, 0.0
        while t < 150 and f in sky.flocks:
            t += 1 / 60
            if f.state == "perch" and px is None:
                px = f.x - 15.0
            dangers = []
            if px is not None:
                px += speed / 60 * (1 if f.x > px else -1)
                dangers = [(px, speed)]
            n0, s0 = sky.takeoff.flights, sky.takeoff.scares
            sky.tick(1 / 60, 100.0, 0.5, dangers)
            if sky.takeoff.flights > n0:
                scares.append(sky.takeoff.scares - s0)
            if px is not None:
                closest = min([closest] + [abs(b.x - px) for b in f.birds if b.state == "perch"])
                if f.state in ("flee", "leave"):
                    break
    sky.takeoff.skill.explore = False
    return d0, sky.takeoff.decide(speed, False)[0], scares, closest


def test_birds_take_off_before_anyone_comes_close_and_learn_to_go_earlier():
    """Замер: к бегущему (3 м/с) расстояние взлёта 2.5 -> 6.4-6.6 м за 80 подходов, испугов из
    первых 10 - 2-4, из последних - 0."""
    d0, d1, scares, closest = approach_runs(1, 60)
    assert d1 > d0 + 2.0 and sum(scares[-10:]) <= sum(scares[:10]) and closest >= CATCH - 0.05


def test_nobody_catches_a_bird():
    """Закон мира: собака и он бегут прямо на сидящих, появляются вплотную - ни разу никто не
    ближе CATCH к сидящей птице."""
    w = Flat()
    sky = sk.Sky(w, 9, "often", np.random.default_rng(3))
    rng = np.random.default_rng(4)
    worst = math.inf
    for rep in range(40):
        sky.flocks = []
        f = sky.spawn(100.0, 0.8)
        t, runner = 0.0, None
        while t < 120 and f in sky.flocks:
            t += 1 / 60
            if f.state == "perch" and runner is None:
                runner = f.x + float(rng.choice([-1, 1])) * float(rng.uniform(0.0, 3.0))    # и вплотную
            dangers = []
            if runner is not None:
                target = min(f.birds, key=lambda b: abs(b.x - runner)).x
                runner += 4.0 / 60 * (1 if target > runner else -1)
                dangers = [(runner, 4.0)]
            sky.tick(1 / 60, 100.0, 0.8, dangers)
            if runner is not None:
                worst = min([worst] + [abs(b.x - runner) for b in f.birds if b.state == "perch"])
    assert worst >= CATCH - 0.05, worst


def test_good_mood_brings_more_and_bigger_flocks_and_off_stays_off():
    for mode in ("rare", "often"):
        assert sk.rate(mode, 0.8) > sk.rate(mode, 0.0) > sk.rate(mode, -0.8)
    assert all(sk.rate("off", m) == 0.0 for m in (-1.0, 0.0, 1.0))
    rng = np.random.default_rng(0)
    sad = [sk.flock_size(-0.8, rng) for _ in range(200)]
    happy = [sk.flock_size(0.9, rng) for _ in range(200)]
    assert max(sad) <= 2 and max(happy) == 8 and np.mean(happy) > np.mean(sad) + 2.0


def test_birds_come_in_the_world_make_sounds_and_the_dog_chases_landed_ones():
    from game import models
    from game.life import Life
    os.environ["AIBOY_HOME"] = tempfile.mkdtemp()
    life = Life(models.load(), 5, 7, 160.0, models.starter_brain(), bird_mode="often")
    life.choice.mood = 0.9
    events = []
    for _ in range(60 * 60 * 6):
        events += [e for e in life.tick() if e[0] in ("birds_came", "sound")]
        if any(e[0] == "birds_came" for e in events):
            break
    assert any(e[0] == "birds_came" for e in events)
    assert sfx.make("flap", 0.6, 0.3) is not None and sfx.make("chirp", 0.5, 0.3) is not None
    dog = life.spawn_dog(near=True)
    f = life.sky.flocks[0]
    for b in f.birds:
        b.state, b.x = "perch", dog.x + 6.0
    f.state = "perch"
    life.dog_ctx.birds = life.sky.perched_x()
    assert "chase" in dog.available(life.dog_ctx)
    off = Life(models.load(), 5, 7, 160.0, models.starter_brain(), bird_mode="off")
    for _ in range(60 * 60 * 3):
        off.tick()
    assert not off.sky.flocks


def test_bird_nets_survive_in_their_world(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    from game import life_store, models
    from game.life import Life
    life = Life(models.load(), 6, 7, 160.0, models.starter_brain(), bird_mode="often")
    fl.practice(life.sky.flyer, life.sky.look, np.random.default_rng(0))
    life.sky.takeoff.flights = 3
    assert life_store.save(life)
    flyer, takeoff = life_store.load_birds(6)
    x = np.zeros(4, np.float32)
    assert np.allclose(flyer.skill.net.forward(x[None]), life.sky.flyer.skill.net.forward(x[None]))
    assert life_store.load_birds(7) is None
