"""Плавание (навык с нуля, безопасность, переправа) и рыбы."""
import numpy as np
import pytest

from acts import swim as swim_act
from body.swim import Stroke, SwimBody
from game import models
from game.life import Life
from skills.swim import Swimmer, try_swim
from world import fish as fishmod
from world import water
from world.generate import generate


@pytest.fixture(scope="module")
def m():
    return models.load()


def deep_world(m, depth=0.7):
    for s in range(7100, 7400):
        w = generate(m.world_genome, s, 160.0)
        if any(p.depth > depth for p in water.ponds(w)):
            return s
    raise AssertionError


def test_flailing_upright_goes_nowhere_and_a_flat_stroke_swims():
    def go(s):
        b = SwimBody(0.0, 0.0)
        for _ in range(360):
            b.step(s)
        return b.path, b.head
    newbie, head0 = go(Stroke(0.95, 2.3, 0.1, 1.2))
    good, head1 = go(Stroke(0.9, 0.7, 1.0, 0.0))
    assert newbie < 3.0 < good and head0 < 0 < head1
    churn, _ = go(Stroke(1.0, 2.4, 0.5, 0.0))           # молотит лёжа: часто и сильно - брызги
    calm, _ = go(Stroke(1.0, 0.9, 0.5, 0.0))
    assert churn < calm


def test_swimming_is_learned_from_zero():
    """Закон обучения: зерно 0, 200 попыток в открытой воде - медиана последних 20 проплывает
    больше медианы первых 20 не меньше чем на 3 м (замер в плане: 1.7-3.9 -> 8.3-8.8 м)."""
    sw = Swimmer(seed=0)
    dist = [try_swim(sw)[1] for _ in range(200)]
    assert np.median(dist[-20:]) > np.median(dist[:20]) + 3.0
    assert sw.can_swim()


def test_he_never_drowns_tired_in_deep_water_he_is_on_shore_within_5_s(m):
    life = Life(m, deep_world(m), 7, 160.0, models.starter_brain())
    p = max(life.water.ponds, key=lambda q: q.depth)
    mid = (p.left + ((p.right - p.left) % 160.0) / 2) % 160.0
    life.human.px = mid
    life.swim.pond = p
    life.swim.enter()
    life.needs.tired = 0.95
    events, t0 = [], life.time
    for _ in range(60 * 8):
        events += life.tick()
        if any(e[0] == "ashore" for e in events):
            break
    assert any(e[0] == "rescued" for e in events)
    ashore = [e for e in events if e[0] == "ashore"]
    assert ashore and life.time - t0 <= swim_act.RESCUE + 0.5
    assert life.water.depth_at(life.human.px) <= life.human.wade_depth()


def test_a_swimmer_crosses_deep_water_on_his_way_a_beginner_does_not(m):
    s = deep_world(m)
    life = Life(m, s, 7, 160.0, models.starter_brain())
    h = life.human
    p = max(life.water.ponds, key=lambda q: q.depth)
    h.px = (p.left - 1.0) % 160.0
    h.py = life.world.height_at(h.px) + h.body.leg()
    h.facing = 1
    assert not h.can_swim and h._blocked(p.left + 0.2, p.left + ((p.right - p.left) % 160) / 2, [(0.0, 0.0)])
    trained = Swimmer(seed=0)
    for _ in range(200):
        try_swim(trained)
    life.swim.swimmer = trained
    h.can_swim = True
    life.walk.goal = (p.right + 2.0) % 160.0           # идёт на ту сторону
    crossed = False
    for _ in range(60 * 60):
        life.tick()
        if life.inv.dist(h.px, (p.right + 2.0) % 160.0) < 1.0:
            crossed = True
            break
    assert crossed


def test_no_swimming_without_deep_water_he_has_seen(m):
    dry = next(s for s in range(7100, 7400) if not water.ponds(generate(m.world_genome, s, 160.0)))
    life = Life(m, dry, 7, 160.0, models.starter_brain())
    assert "swim" not in life.available()


def test_fish_look_comes_from_the_fish_network_and_they_stay_in_the_water(m):
    life = Life(m, deep_world(m), 7, 160.0, models.starter_brain())
    assert life.fish.fish
    assert len({fishmod.look(life.world.seed, 0, k) for k in range(5)}) == 5     # рыбы пруда разные
    assert fishmod.look(5, 0, 0) == fishmod.look(5, 0, 0) != fishmod.look(6, 0, 0)
    for _ in range(60 * 20):
        life.fish.step(1 / 60)
    for f in life.fish.fish:
        x, y = life.fish.position(f)
        assert life.water.depth_at(x) > 0 and y < life.water.ponds[f.pond].level


def test_fish_bite_more_at_dawn_and_dusk_and_at_their_depth(m):
    life = Life(m, deep_world(m), 7, 160.0, models.starter_brain())
    sc = life.fish
    f = sc.fish[0]
    dusk = sc.bite_chance(0.25 * fishmod.DAY, f.depth, f, 1.0)
    noon = sc.bite_chance(0.5 * fishmod.DAY, f.depth, f, 1.0)
    off = sc.bite_chance(0.25 * fishmod.DAY, f.depth + 0.6, f, 1.0)
    assert dusk > noon > 0 and dusk > 5 * off


def test_swimming_skill_lives_with_the_person(tmp_path):
    from skills import store
    sw = Swimmer(seed=3)
    for _ in range(30):
        try_swim(sw)
    assert store.save(55, "swim", sw)
    again = store.load(55, "swim", lambda: Swimmer(seed=3))
    assert again.attempts == 30 and again.skill == sw.skill and again.can_swim() == sw.can_swim()
