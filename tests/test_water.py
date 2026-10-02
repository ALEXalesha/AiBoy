"""Вода: пруды в котловинах (сеть воды), уровень не выше перелива, не больше одного глубокого;
водная физика для предметов."""
import math

import numpy as np
import pymunk
import pytest

from game import models
from physics import water as pwater
from physics.world import PhysWorld
from world import water
from world.generate import generate


@pytest.fixture(scope="module")
def m():
    return models.load()


@pytest.fixture(scope="module")
def worlds(m):
    return [generate(m.world_genome, 7100 + i, (80.0, 160.0, 320.0, 640.0)[i % 4]) for i in range(40)]


def test_water_is_never_above_its_spill_height_and_lies_in_basins(worlds):
    for w in worlds:
        for p in water.ponds(w):
            assert p.bottom < p.level <= p.spill + 1e-9
            xs = np.linspace(p.left, p.left + (p.right - p.left) % w.width, 20)
            assert all(w.height_at(x) <= p.level + 0.05 for x in xs[1:-1])     # внутри - ниже воды


def test_at_most_one_deep_pond_so_the_ring_stays_walkable(worlds):
    for w in worlds:
        assert sum(p.depth > water.WADE for p in water.ponds(w)) <= 1


def test_about_sixty_percent_of_worlds_have_water_one_to_four_ponds(worlds, m):
    assert models.passport("water").get("generations", 0) >= 20
    from train.water import max_ponds
    more = worlds + [generate(m.world_genome, 7300 + i, (80.0, 160.0, 320.0, 640.0)[i % 4]) for i in range(80)]
    share = np.mean([bool(water.ponds(w)) for w in more])
    # цель спецификации - около 60 %; эволюция дала 0.43-0.53 на пяти наборах по 40 миров
    assert 0.4 <= share <= 0.75, share
    wet = [water.ponds(w) for w in worlds]
    counts = [len(ps) for ps in wet if ps]
    assert np.median(counts) <= 4
    over = sum(len(ps) > max_ponds(w.width) for w, ps in zip(worlds, wet))
    assert over <= 0.1 * len(worlds), over          # замер после эволюции: 2 из 40


def test_water_lookups(worlds):
    w = next(w for w in worlds if water.ponds(w))
    wt = water.Water(w)
    p = wt.ponds[0]
    mid = (p.left + ((p.right - p.left) % w.width) / 2) % w.width
    assert wt.pond_at(mid) is p and wt.depth_at(mid) > 0
    dry = next(x for x in np.arange(0, w.width, 0.5) if wt.pond_at(x) is None)
    assert wt.depth_at(dry) == 0.0 and wt.level_at(dry) is None


class Flat:
    width, dx = 100.0, 0.25

    def height_at(self, x):
        return -3.0


@pytest.mark.parametrize("density", [300.0, 600.0])
def test_a_body_floats_at_its_share(density):
    """Плотность 300 - под водой 30 %, 600 - 60 % (Архимед), после того как успокоится."""
    p = PhysWorld(Flat(), center=0.0)
    r = 0.2
    mass = density * math.pi * r * r
    p.add_ball("бревно", 0.0, 0.5, radius=r, mass=mass)
    b = p.body("бревно")
    share = [0.0]
    p.forces.append(lambda dt: share.__setitem__(0, pwater.apply(b, "circle", r, 0.0, dt)))
    for _ in range(60 * 8):
        p.step(1 / 60)
    share = share[0]
    assert share == pytest.approx(density / pwater.RHO, abs=0.05)
    assert abs(b.velocity.y) < 0.05


def test_a_dense_body_sinks_and_water_slows_things():
    p = PhysWorld(Flat(), center=0.0)
    p.add_box("камень", 0.0, 0.0, 0.2, 0.2, mass=2500 * 0.04)
    b = p.body("камень")
    p.forces.append(lambda dt: pwater.apply(b, "box", (0.2, 0.2), 0.0, dt))
    for _ in range(60 * 5):
        p.step(1 / 60)
    assert b.position.y < -2.5                                    # на дне
    q = PhysWorld(Flat(), center=0.0)
    q.add_ball("мяч", 0.0, -0.5, radius=0.11, mass=0.4)
    bq = q.body("мяч")
    bq.velocity = (5.0, 0.0)
    q.forces.append(lambda dt: pwater.apply(bq, "circle", 0.11, 0.0, dt))
    for _ in range(30):
        q.step(1 / 60)
    assert abs(bq.velocity.x) < 2.0


def test_a_fast_entry_splashes_and_the_waves_fade():
    w = pwater.Waves(-5.0, 5.0)
    body = pymunk.Body(1.0, 1.0)
    body.position = (0.0, 0.05)
    body.velocity = (0.0, -4.0)
    pwater.apply(body, "circle", 0.1, 0.0, 1 / 60, waves=w)
    for _ in range(10):
        w.step(1 / 60)
    peak = float(np.abs(w.y).max())
    assert peak > 0.001
    for _ in range(60 * 10):
        w.step(1 / 60)
    assert float(np.abs(w.y).max()) < peak / 5


# --- он и вода ---

def wet_world(m, deep=True):
    for s in range(7100, 7300):
        w = generate(m.world_genome, s, 160.0)
        ps = water.ponds(w)
        if ps and (not deep or any(p.depth > 0.6 for p in ps)):
            return s
    raise AssertionError("нет мира с водой")


def test_he_appears_on_dry_land(m):
    from game.life import Life
    for s in range(7100, 7160):
        life = Life(m, s, 7, 80.0, models.starter_brain())
        assert life.water.depth_at(life.human.px) == 0.0
        assert life.water.depth_at(life.inv.chest_x) == 0.0


def test_deep_water_stops_him_until_he_can_swim_and_shallow_water_slows_him(m):
    from game.life import Life
    life = Life(m, wet_world(m), 7, 160.0, models.starter_brain())
    h = life.human
    p = max(life.water.ponds, key=lambda q: q.depth)
    deepest = p.left + ((p.right - p.left) % 160.0) / 2
    shore = p.left - 0.3
    rel = [(0.0, 0.0)]
    assert h._blocked(shore, deepest, rel) or h.water.depth_at(deepest) <= h.wade_depth()
    h.can_swim = True
    assert not h._blocked(shore, shore + 0.25, rel) or h._req(shore + 0.25, rel) - h._req(shore, rel) > 0.03


def test_he_walks_to_a_pond_he_has_seen_to_drink_and_fills_the_flask(m):
    from game.life import Life
    life = Life(m, wet_world(m, deep=False), 7, 160.0, models.starter_brain())
    p = life.water.ponds[0]
    for d in range(-12, 12):                      # был у пруда
        life.walk.places.visit(p.left + d * 0.5, 0.0, 1.0)
    life.human.px = (p.left - 6.0) % 160.0
    life.human.py = life.world.height_at(life.human.px) + life.body.leg()
    life.backpack.flask = 0.0
    life.needs.thirst = 0.95
    events = []
    for _ in range(60 * 60):
        events += life.tick()
        if any(e[0] == "drank" for e in events):
            break
    drank = [e for e in events if e[0] == "drank"]
    assert drank and drank[0][1] == "вода" and life.needs.thirst < 0.1 and life.backpack.flask == 1.0
