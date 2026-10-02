"""Удочка, велосипед, фломастер; все игрушки - в реестре и в жизни."""
import math

import numpy as np
import pytest

import toys.bicycle as Bi
import toys.marker as Mk
import toys.rod as R
from game import models
from game.life import Life
from physics.world import PhysWorld
from sound import sfx
from toys import registry
from world.fish import School
from world.generate import generate
from world.water import Water


@pytest.fixture(scope="module")
def pond():
    m = models.load()
    s = next(s for s in range(7100, 7400) if any(p.depth > 0.7 and ((p.right - p.left) % 160) > 5
                                                 for p in Water(generate(m.world_genome, s, 160.0)).ponds))
    w = generate(m.world_genome, s, 160.0)
    wt = Water(w)
    p = max(wt.ponds, key=lambda q: q.depth)
    return w, wt, School(w, wt), (p.left - 1.0) % 160


def test_fishing_is_learned(pond):
    """Закон обучения: зерно 0, 500 забросов; доля пойманных за последние 100 выше первых 100
    (замер: 0.05-0.14 -> 0.25-0.34)."""
    w, wt, sc, x = pond
    h = R.RodHeads(seed=0)
    rng = np.random.default_rng(0)
    out = [R.practice(h, rng, w, wt, sc, x, t0=i * 30.0) for i in range(500)]
    assert np.mean(out[-100:]) > np.mean(out[:100]) + 0.1


def test_a_cast_onto_dry_land_catches_nothing_and_no_rod_without_water():
    from toys.blocks import Flat
    from toys.inplace import Ctx, run
    h = R.RodHeads(seed=1)
    success, entry = run(R.RodAttempt(h, Ctx(Flat(), 0.0, rng=np.random.default_rng(1))))
    assert success == 0.0 and entry["why"] == "мимо воды"
    m = models.load()
    dry = next(s for s in range(7100, 7400) if not Water(generate(m.world_genome, s, 160.0)).ponds)
    life = Life(m, dry, 7, 160.0, models.starter_brain())
    assert "rod" not in life.available()


def test_the_bicycle_rolls_straight_on_flat_ground():
    p = PhysWorld(Bi.Flat(), 0.0)
    b = Bi.Bike(p, 0.0, Bi.WHEEL_R + 0.05, 1)
    b.control(0.5, 0.5)
    worst = 0.0
    for _ in range(60 * 8):
        p.follow(b.x)
        p.step(1 / 60)
        worst = max(worst, abs(b.angle()))
    assert b.x > 15.0 and worst < 0.2


def test_riding_over_hills_is_learned():
    """Закон обучения: зерно 0, 300 поездок по холмам; медиана последних 50 выше первых 50
    (замер: 0.01-0.05 -> 0.87-1.0, новичок встаёт на дыбы)."""
    h = Bi.BikeHeads(seed=0)
    rng = np.random.default_rng(0)
    s = [Bi.practice(h, rng) for _ in range(300)]
    assert np.median(s[-50:]) > np.median(s[:50]) + 0.4


def test_drawing_taste_and_learning():
    t = np.linspace(0, 2 * math.pi, 25)
    circle = np.c_[0.3 * np.cos(t), 0.3 * np.sin(t)]
    line = np.c_[np.linspace(0, 1.4, 25), np.zeros(25)]
    taste = Mk.DrawTaste(0)
    pens = np.zeros(24)
    assert taste.beauty(Mk.features(circle, pens)) > taste.beauty(Mk.features(line, pens)) + 0.2
    h = Mk.MarkerHeads(seed=0)
    s = [Mk.practice(h) for _ in range(300)]
    assert np.median(s[-50:]) > np.median(s[:50]) + 0.05          # замер: 0.82 -> 0.96


def test_owner_ratings_teach_the_drawing_taste():
    taste = Mk.DrawTaste(1)
    t = np.linspace(0, 2 * math.pi, 25)
    wide = Mk.features(np.c_[0.5 * np.cos(t), 0.2 * np.sin(t)], np.zeros(24))
    tall = Mk.features(np.c_[0.2 * np.cos(t), 0.5 * np.sin(t)], np.zeros(24))
    for _ in range(10):
        taste.rate(wide, -1)
        taste.rate(tall, 1)
    assert taste.beauty(tall) > taste.beauty(wide) or taste.beauty(tall) == 1.0


def test_every_toy_is_registered_has_heads_and_a_sound():
    names = registry.names()
    for n in ("guitar", "ball", "blocks", "bubbles", "plane", "boomerang", "rod", "bicycle", "marker"):
        assert n in names
    for n in registry.INPLACE:
        heads_cls, attempt_cls = registry.inplace(n)
        assert heads_cls(seed=0) is not None and hasattr(attempt_cls, "step")
    for ev in ("scratch", "chain", "splash", "clack", "whoosh", "pop"):
        assert sfx.make(ev, 0.5, 0.1) is not None


@pytest.mark.parametrize("name", ["blocks", "bubbles", "plane", "boomerang", "bicycle", "marker"])
def test_he_plays_each_toy_in_the_world(name):
    m = models.load()
    life = Life(m, 5, 7, 160.0, models.starter_brain())
    others = [n for n in registry.names() if n not in ("guitar", name)]
    events = []
    for _ in range(60 * 240):
        c = life.choice
        for n in others:
            c.acts[n].liking = 0.0
        c.acts[name].liking, c.acts["explore"].liking, c.acts[name].boredom = 1.0, 0.0, 0.0
        events += life.tick()
        if any(e[0] == "toy_attempt" for e in events):
            break
    done = [e for e in events if e[0] == "toy_attempt"]
    assert done and done[0][1] == name and 0.0 <= done[0][2]["success"] <= 1.0
    assert life.toy_journals[name].entries
