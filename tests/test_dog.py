"""Собака (спецификация 2.0, раздел 8): вид и нрав из сети, тело, движение учится, доверие,
приручение, мяч, своя скука, нужды, вода, не застревает, появление по настроению."""
import math

import numpy as np
import pytest

from animals.dog import motor as M
from animals.dog import practice as P
from animals.dog import spawn
from animals.dog.body import DogBody
from animals.dog.dog import Dog, DogCtx
from animals.dog.look import look
from game import models
from physics.world import PhysWorld
from world.generate import generate
from world.water import Water


def test_look_and_temper_come_from_the_dog_network_by_world_seed():
    a, b = look(1), look(2)
    assert a == look(1) and a != b
    many = [look(s) for s in range(60)]
    for k in ("size", "ears", "tail", "shy", "playful", "stamina"):
        v = [getattr(d, k) for d in many]
        assert 0.0 <= min(v) and max(v) <= 1.0 and max(v) - min(v) > 0.5, k
    assert len({d.color for d in many}) == 60


def test_the_body_stands_still_and_trots_straight_on_flat_ground():
    p = PhysWorld(P.Flat(), 50.0)
    d = DogBody(p, 50.0, look(0))
    d.pose = "stand"
    y0 = d.y
    for _ in range(180):
        d.step(1 / 60)
        p.step(1 / 60)
    assert abs(d.y - y0) < 0.06 and abs(d.x - 50.0) < 0.05 and abs(d.angle()) < 0.05
    d.pose, d.gait = None, (2.4, 0.35, 0.25, 0.6)
    worst = 0.0
    for _ in range(60 * 8):
        p.follow(d.x)
        d.step(1 / 60)
        p.step(1 / 60)
        worst = max(worst, abs(d.angle()))
    assert d.x - 50.0 > 6.0 and worst < 0.3


def test_it_climbs_hills_both_ways_without_flipping():
    m = models.load()
    w = generate(m.world_genome, 900, 160.0)
    for facing in (1, -1):
        p = PhysWorld(w, 10.0)
        d = DogBody(p, 10.0, look(0), facing)
        d.gait = (2.4, 0.35, 0.25, 0.6)
        worst = 0.0
        for _ in range(60 * 30):
            p.follow(d.x)
            d.step(1 / 60)
            p.step(1 / 60)
            worst = max(worst, M.tilt(w, d))
        assert facing * (d.x - 10.0) > 15.0 and worst < 1.0, (facing, d.x, worst)


@pytest.mark.parametrize("seed", [1, 2])
def test_moving_is_learned(seed):
    """Закон обучения движения: новичок с привычкой по зерну (семенит, запинается); сеть без шума
    на 8 одинаковых забегах до и после 100 забегов учёбы. Замер на ровном (план): 0.23-1.0 ->
    1.21-1.23 у зёрен 0-3; на холмах награда больше от склона, чем от походки (там закон -
    «не застревает»)."""
    mot = M.DogMotor(seed=seed)
    w = P.Flat()

    def judge():
        mot.skill.explore = False
        r = [M.practice(mot, w, 20.0 + 20 * i, np.random.default_rng(i), learn=False) for i in range(8)]
        mot.skill.explore = True
        return float(np.mean(r))
    before = judge()
    rng = np.random.default_rng(seed)
    for _ in range(100):
        M.practice(mot, w, float(rng.uniform(40, 160)), rng)
    after = judge()
    assert after > before + 0.2 and after > 1.0, (before, after)


@pytest.mark.parametrize("world_seed", [5, 9])
def test_it_gets_used_to_him_when_he_is_patient_and_flees_when_he_runs_at_it(world_seed):
    """Замер (план): терпеливо за 4 минуты ручными стали две пугливостью 0.14 и 0.22 из трёх
    (0.41 - доверие 0.16); бегом на неё (1.5 м/с) доверие 0.4 -> 0 и она убегает."""
    calm = P.approach(world_seed, patient=True)
    rush = P.approach(world_seed, patient=False, trust=0.4)
    assert calm["trust"][-1] > 0.5 and calm["trust"][-1] > calm["trust"][0]
    assert rush["trust"][-1] < 0.1 and rush["fled"] >= 1 and rush["away_speed"] > 0.2


def test_a_tamed_dog_brings_the_ball_more_often_after_throws_and_gets_bored_of_it():
    """Сеть «принести» сначала чаще убегает с мячом (привычка новичка); по награде «похвалили и
    дали лакомство» за 60 бросков учится нести. И сама бросает приносить (скука) и потом
    возвращается к мячу - доверие от этого не падает."""
    r = P.fetch(5, throws=60, needs=False)               # нужды на нуле: уходит от мяча от скуки
    g = r["grabs"]
    half = len(g) // 2
    assert r["before"] < 0.25 and r["after"] > 0.6 and r["brought"] >= 3, g
    assert g[half:].count("bring") / max(1, len(g) - half) > g[:half].count("bring") / max(1, half), g
    away = [i for i, s in enumerate(r["switches"]) if s[0] == "fetch"]
    back = [i for i, s in enumerate(r["switches"]) if s[1] == "fetch"]
    assert away and any(b > away[0] for b in back), r["switches"]
    assert r["dog"].trust >= 0.9


def test_without_praise_it_does_not_learn_to_bring():
    r = P.fetch(5, throws=40, praise=False)
    assert r["after"] < 0.3, (r["before"], r["after"], r["grabs"])


class Cliff:
    """Ровно, а с 60 м - стена 2.5 м: не запрыгнуть."""
    width, dx = 200.0, 0.25

    def height_at(self, x):
        return 0.0 if (x % self.width) < 60.0 else 2.5


def test_at_a_wall_it_does_not_push_forever():
    """Упёрлась в стену, на которую не запрыгнуть, - прыгает, потом идёт другой дорогой."""
    w = Cliff()
    p = PhysWorld(w, 52.0)
    dog = Dog(3, p, 52.0)
    ctx = DogCtx(w, p)
    dog.wander_dir, dog.wander_t = 1, 1e9                # своей волей не повернёт
    xs = []
    for k in range(60 * 30):
        P._step(dog, p, ctx)
        xs.append(dog.x)
    assert max(xs) < 60.5 and xs[-1] < 56.0 and dog.stuck_turns >= 1, (max(xs), xs[-1])


@pytest.mark.parametrize("world_seed", [900, 901, 902, 903, 904, 905])
def test_it_never_gets_stuck(world_seed):
    """Гуляет одна 3 минуты по холмам (и воде, где она есть): ни разу 10 с подряд не хотела идти
    без продвижения на метр - прыгает, а не выйдет - идёт другой дорогой."""
    m = models.load()
    w = generate(m.world_genome, world_seed, 160.0)
    r = P.wander(world_seed, w, minutes=3.0, water=Water(w))
    assert r["worst_stuck"] <= 10.0 and r["walked"] > 30.0, r


def test_it_sits_and_lies_when_tired_and_rests():
    w = P.Flat()
    p = PhysWorld(w, 50.0)
    dog = Dog(3, p, 50.0)
    ctx = DogCtx(w, p)
    poses = []
    dog.tired = 0.9
    dog.choice.since, dog.decide_t = 999.0, 99.0
    for _ in range(60 * 10):
        P._step(dog, p, ctx)
        if dog.choice.current == "rest" and (not poses or poses[-1] != dog.pose):
            poses.append(dog.pose)
    assert dog.choice.current == "rest" and dog.tired < 0.9
    assert poses[:2] == ["sit", "lie"], poses           # садится, потом ложится
    for _ in range(60 * 60):
        P._step(dog, p, ctx)
    assert dog.tired < 0.6                              # отдохнула


def test_it_swims_in_deep_water_and_shakes_off_on_the_shore():
    m = models.load()
    seed = next(s for s in range(7100, 7400) if any(q.depth > 0.9 for q in Water(generate(m.world_genome, s, 160.0)).ponds))
    w = generate(m.world_genome, seed, 160.0)
    wt = Water(w)
    pond = max(wt.ponds, key=lambda q: q.depth)
    assert pond.left < pond.right
    start = pond.left - 1.5
    p = PhysWorld(w, start)
    dog = Dog(seed, p, start)
    ctx = DogCtx(w, p, wt)
    dog.choice.current, dog.choice.since = "explore", 0.0
    dog.wander_dir, dog.wander_t = 1, 60.0           # идёт вправо - через пруд
    afloat, swam, shook, t_out = True, 0.0, False, None
    for k in range(60 * 40):
        P._step(dog, p, ctx)
        hx, hy = dog.body.head()
        level = wt.level_at(dog.x)
        if level is not None and dog.body.swimming:
            swam += 1 / 60
            if swam > 1.5:                           # прыгнула с берега - окунулась, потом плывёт
                afloat = afloat and hy > level
        shook = shook or (swam > 0 and dog.body.shake > 0)
        if t_out is None and swam > 0 and dog.x > pond.right + 1.0:
            t_out = k / 60
    assert swam > 1.0 and afloat and t_out is not None and t_out < 30.0 and shook, (swam, afloat, t_out, shook)


def test_it_eats_when_hungry_and_drinks_at_the_pond():
    m = models.load()
    seed = next(s for s in range(7100, 7400) if Water(generate(m.world_genome, s, 160.0)).ponds)
    w = generate(m.world_genome, seed, 160.0)
    wt = Water(w)
    p = PhysWorld(w, 80.0)
    dog = Dog(seed, p, 80.0)
    ctx = DogCtx(w, p, wt)
    dog.hunger, dog.thirst = 0.95, 0.0
    dog.choice.since, dog.decide_t = 999.0, 99.0
    for _ in range(60 * 20):
        P._step(dog, p, ctx)
    assert dog.hunger < 0.6
    dog.thirst = 0.95
    dog.choice.since, dog.decide_t = 999.0, 99.0
    for _ in range(60 * 150):
        P._step(dog, p, ctx)
        if dog.thirst < 0.3:
            break
    assert dog.thirst < 0.3


def test_good_mood_brings_the_dog_more_likely_and_never_stays_never():
    for mode in ("sometimes", "often"):
        assert spawn.world_chance(mode, 0.8) > spawn.world_chance(mode, 0.0) > spawn.world_chance(mode, -0.8)
        assert spawn.minute_chance(mode, 0.8) > spawn.minute_chance(mode, -0.8)
    assert 0.3 < spawn.world_chance("sometimes", 0.0) < 0.5
    for mood in (-1.0, 0.0, 1.0):
        assert spawn.world_chance("never", mood) == 0.0 and spawn.minute_chance("never", mood) == 0.0
    rng = np.random.default_rng(0)
    assert spawn.stay_minutes(0.8, rng) > spawn.stay_minutes(0.0, np.random.default_rng(0))


def test_dog_state_survives_as_dict_and_broken_values_are_ignored():
    p = PhysWorld(P.Flat(), 50.0)
    a = Dog(4, p, 50.0)
    a.trust, a.tamed, a.hunger = 0.8, True, 0.3
    a.choice.acts["fetch"].boredom = 0.4
    b = Dog(4, p, 60.0).load(a.to_dict())
    assert (b.trust, b.tamed, b.hunger, b.choice.acts["fetch"].boredom) == (0.8, True, 0.3, 0.4)
    c = Dog(4, p, 70.0).load({"trust": 9, "tamed": "yes", "hunger": None, "choice": {"fetch": {"liking": -1}}})
    assert c.trust == 0.0 and c.tamed is False and c.hunger == 0.2


def test_taming_is_learned():
    """Закон обучения головы «приручать»: 300 попыток с новой неручной собакой мира 5; медиана
    последних 50 выше первых 50 (замер: зерно 0 - 0.12 -> 0.55, зерно 1 - 0.39 -> 0.56). Новичок
    подходит быстро (он ходит 2 м/с - собака видит бег на себя) и протягивает лакомство издали
    ближе, чем она подпускает."""
    from animals.dog.session import TameHeads, tame_practice
    h = TameHeads(seed=0)
    rng = np.random.default_rng(0)
    s = [tame_practice(h, 5, rng) for _ in range(300)]
    assert np.median(s[-50:]) > np.median(s[:50]) + 0.2
