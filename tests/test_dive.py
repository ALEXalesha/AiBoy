"""Маска для ныряния (спецификация 2.0, раздел 5): с ней дольше под водой и всё равно не тонет;
без неё - прежний предел; берёт её только для долгого плавания; нырять учится."""
import os
import tempfile

import numpy as np
import pytest

from acts import dive
from acts import swim as sw
from game import models
from game.life import Life
from sound import sfx
from world.generate import generate
from world.water import Water


@pytest.mark.parametrize("seed", [0, 2])
def test_diving_with_the_mask_is_learned(seed):
    """Закон обучения головы маски: 800 нырков; доля «увидел рыбу» за последние 100 выше первых 100
    (замер: зёрна 0 / 1 / 2 - 0.39 -> 0.69, 0.35 -> 0.43, 0.34 -> 0.62)."""
    h = dive.MaskHeads(seed=seed)
    rng = np.random.default_rng(seed)
    s = [dive.practice(h, rng) for _ in range(800)]
    assert np.mean(s[-100:]) > np.mean(s[:100]) + 0.15


@pytest.fixture(scope="module")
def pond_world():
    m = models.load()
    seed = next(s for s in range(7100, 7400) if any(q.depth > 1.2 for q in Water(generate(m.world_genome, s, 160.0)).ponds))
    return m, seed


def in_water(m, seed, mask):
    os.environ["AIBOY_HOME"] = tempfile.mkdtemp()
    life = Life(m, seed, 7, 160.0, models.starter_brain())
    pond = max(life.water.ponds, key=lambda q: q.depth)
    h = life.human
    h.px = (pond.left + pond.right) / 2
    h.py = life.world.height_at(h.px) + h.body.leg()
    life.swim.pond = pond
    life.swim.enter()
    life.swim.swimmer.history = [2.0] * 10                 # плавать умеет - смотрим на маску
    life.choice.current = "dive" if mask else "swim"
    if mask:
        life.swim.put_on_mask(dive.MaskHeads(seed=1))
    return life


def under_run(life, seconds):
    """(самый долгий отрезок с головой под водой, был ли на берегу, события)."""
    longest = cur = 0.0
    events = []
    for _ in range(int(seconds * 60)):
        life.needs.hunger = life.needs.thirst = 0.0
        events += life.tick()
        if life.swim.in_water and life.swim.body.head <= 0.0:
            cur += 1 / 60
            longest = max(longest, cur)
        else:
            cur = 0.0
        if not life.swim.in_water:
            break
    return longest, not life.swim.in_water, events


def test_with_the_mask_he_stays_under_much_longer_and_still_never_drowns(pond_world):
    m, seed = pond_world
    life = in_water(m, seed, mask=True)
    longest, _, events = under_run(life, 120)
    assert longest > 3 * sw.UNDER and longest <= dive.AIR + 0.5, longest
    assert any(e[0] == "dive" for e in events) and ("sound", "bubbles", 0.5, 0.05) in events
    # держаться хочет дольше, чем хватит воздуха, - всплывает, пока воздуха хватает на подъём
    life = in_water(m, seed, mask=True)
    life.swim.mask_heads.plan = lambda *a: (2.0, 60.0, None)
    life.swim.mask_heads.finish = lambda *a, **k: 0.0
    lowest = dive.AIR
    for _ in range(60 * 90):
        life.needs.hunger = life.needs.thirst = life.needs.tired = 0.0
        life.tick()
        lowest = min(lowest, life.swim.air)
    assert lowest >= 0.0
    # устал - всё та же безопасность: на берегу, маску снял
    life = in_water(m, seed, mask=True)
    life.needs.tired = sw.TIRED_OUT - 0.01
    _, ashore, events = under_run(life, 30)
    assert ashore and any(e[0] == "rescued" for e in events) and life.swim.mask is False


def test_without_the_mask_the_old_limit_holds(pond_world):
    m, seed = pond_world
    life = in_water(m, seed, mask=False)
    longest, _, _ = under_run(life, 60)
    assert longest <= sw.UNDER + 0.1


def test_he_takes_the_mask_only_when_he_wants_a_long_swim(pond_world):
    m, seed = pond_world
    os.environ["AIBOY_HOME"] = tempfile.mkdtemp()
    for want, expect in (("dive", True), ("swim", False)):
        life = Life(m, seed, 7, 160.0, models.starter_brain())
        pond = max(life.water.ponds, key=lambda q: q.depth)
        life.walk.places.visits[:] = 1                       # пруд он видел
        c = life.choice
        for n in c.acts:
            c.acts[n].liking = 0.0
        c.acts[want].liking = 1.0
        masked, events = False, []
        for _ in range(60 * 240):
            life.needs.hunger = life.needs.thirst = life.needs.tired = 0.0
            events += life.tick()
            masked = masked or life.swim.mask
            if masked:
                break
        assert masked is expect, (want, [e[0] for e in events if e[0] in ("mask_on", "pickup_toy")])
        assert pond is not None


def test_the_diving_sound_exists():
    assert sfx.make("bubbles", 0.5, 0.05) is not None
