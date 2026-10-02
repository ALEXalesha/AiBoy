"""Нужды, еда, рюкзак, отдых (задача 9 плана 2.0)."""

import pytest

from acts import needs as nd

from acts.eat import FoodPrefs
from acts.needs import Backpack, Needs
from game import models
from game.life import Life
from world import food
from world.generate import generate


@pytest.fixture(scope="module")
def m():
    return models.load()


def test_needs_grow_hunger_slowly_thirst_faster_and_tiredness_with_the_way():
    n = Needs()
    n.tick(60.0)
    assert 0 < n.hunger < n.thirst and n.tired == 0.0
    n.tick(1.0, walked=100.0, climbed=10.0)
    assert n.tired > 0.03
    day, night = Needs(), Needs()
    day.tick(60.0, night=False)
    night.tick(60.0, night=True)
    assert day.thirst > night.thirst                          # солнце


def test_lying_rests_faster_than_sitting_and_tiredness_shortens_the_stride():
    sit, lie = Needs(), Needs()
    sit.tired = lie.tired = 0.8
    sit.tick(30.0, posture="sit")
    lie.tick(30.0, posture="lie")
    assert lie.tired < sit.tired < 0.8
    n = Needs()
    assert n.stride() == 1.0
    n.tired = 1.0
    assert n.stride() == pytest.approx(nd.SLOW_MIN)
    n.tired = 0.4
    assert n.stride() == 1.0


def test_urgency_grows_with_the_need_and_is_zero_when_fine():
    n = Needs()
    assert n.urgency() == {"eat": 0.0, "drink": 0.0, "rest": 0.0}
    n.hunger = 0.6
    a = n.urgency()["eat"]
    n.hunger = 0.95
    assert 0 < a < n.urgency()["eat"]


def test_food_taste_comes_from_a_network_by_world_seed_some_good_some_bad(m):
    w = generate(m.world_genome, 11, 320)
    f = food.Food(w)
    assert f.items
    tastes = [food.taste(w.seed, f.feats[i]) for i, _ in f.items]
    assert min(tastes) < -0.2 and max(tastes) > 0.2
    assert tastes == [food.taste(w.seed, f.feats[i]) for i, _ in f.items]
    other = [food.taste(12, f.feats[i]) for i, _ in f.items]
    assert tastes != other


def test_a_food_he_found_bad_is_chosen_less_often_afterwards():
    prefs = FoodPrefs(seed=0)
    bad = food.features("bush", (0.8, 0.1, 0.2), 0.8)
    good = food.features("tree", (0.9, 0.7, 0.1), 1.0)
    options = [("плохая", bad, 5.0), ("хорошая", good, 5.0)]
    before = prefs.probabilities(options)["плохая"]
    prefs.learn(bad, -0.9)
    prefs.learn(good, 0.8)
    after = prefs.probabilities(options)["плохая"]
    assert before > 0.2 and after < before / 2


def test_regrow_and_backpack_refill_at_the_chest(m):
    life = Life(m, 5, 7, 160.0, models.starter_brain())
    f = life.food
    i, e = f.items[0]
    f.eat(i, 0.0)
    assert all(j != i for j, _ in f.available(1.0)) and any(j == i for j, _ in f.available(food.REGROW + 1))
    life.backpack.food = 0
    life.backpack.flask = 0.0
    life.human.px = life.inv.chest_x
    life.tick()
    assert life.backpack.full()


@pytest.mark.parametrize("tired,posture", [(0.6, "sit"), (0.95, "lie")])
def test_tiredness_makes_him_sit_and_more_tiredness_lie(m, tired, posture):
    life = Life(m, 5, 7, 160.0, models.starter_brain())
    for _ in range(60):
        life.tick()
    life.needs.tired = tired
    life.choice.switch("rest")
    life.daily.start("rest", life.human.px, life.time)
    for _ in range(60):
        life.tick()
    assert life.posture == posture
    if posture == "lie":
        assert life.human.fallen > 0                   # лёг
    else:
        assert life.human.angles[6] > 1.2             # ноги вперёд - сидит на земле


def test_he_eats_when_hungry_and_says_how_it_tasted(m):
    life = Life(m, 5, 7, 160.0, models.starter_brain())
    life.needs.hunger = 0.95
    events = []
    for _ in range(60 * 90):
        events += life.tick()
        if any(e[0] == "ate" for e in events):
            break
    ate = [e for e in events if e[0] == "ate"]
    assert ate and life.needs.hunger < 0.95 - food.PORTION + 0.05
    i = next(k for k, e in enumerate(events) if e[0] == "ate")
    assert any(e[0] == "phrase" for e in events[i:])


def worst_streak(life, minutes):
    """Самое долгое время подряд (с), когда хоть одна нужда выше 0.9."""
    streak = best = 0.0
    for _ in range(int(minutes * 3600)):
        life.tick()
        if life.needs.worst() > 0.9:
            streak += 1 / 60
            best = max(best, streak)
        else:
            streak = 0.0
    return best


def test_no_need_stays_above_0_9_for_more_than_a_minute(m):
    """Закон спецификации на 5 человечках: 20 минут жизни, нужды с высокой точки."""
    from tools.measure_sessions import BODIES
    for b in BODIES:
        life = Life(m, 1000 + b, b, 160.0, models.starter_brain())
        life.needs.hunger, life.needs.thirst, life.needs.tired = 0.85, 0.6, 0.7
        assert worst_streak(life, 20) <= 60.0, b


def test_needs_and_backpack_survive_as_dicts_and_broken_values_are_ignored():
    n = Needs().load({"hunger": 0.4, "thirst": 2.0, "tired": "x"})
    assert (n.hunger, n.thirst, n.tired) == (0.4, 0.0, 0.0)
    b = Backpack().load({"food": 1, "flask": 0.5, "treats": 99})
    assert (b.food, b.flask, b.treats) == (1, 0.5, nd.TREATS)


