"""Навигатор - голова «гулять»: сам выбирает, куда идти.

Замер до (1.1.0, план 2.0): в 7 мирах из 10 за 12 минут 0-4 разворота, в двух - дёрганье
(16-24 «туда-сюда»): любопытство не помнит мест, и развернуться никогда не выгодно.
"""
import numpy as np
import pytest

from acts.explore import navigator as navmod
from acts.explore.navigator import BACK, FORWARD, LOOK, Navigator, Walk
from acts.policy import ChoicePolicy
from game import models, probe

MAX_PER_MIN = 3.0          # разворотов в минуту в изученном мире - не больше (мир 80 м он обходит за минуту)
EXPLORED_WORLDS = [(11, 80), (12, 80), (20, 80), (21, 80), (22, 80), (23, 80), (24, 80), (25, 80), (26, 80),
                   (27, 80)]


def test_choice_policy_learns_the_better_action_and_keeps_the_others_possible():
    pol = ChoicePolicy(3, 3, seed=0, lr=1e-2)
    x = np.array([1.0, 0.0, 0.5])
    before = pol.probs(x)[2]
    rng = np.random.default_rng(0)
    for _ in range(600):
        a, _ = pol.choose(x, rng=rng)
        pol.learn(x, None, a, 1.0 if a == 2 else 0.0)
    p = pol.probs(x)
    assert p[2] > max(0.8, before + 0.3) and p.min() > 1e-4


def test_masked_action_is_never_chosen():
    pol = ChoicePolicy(2, 3, seed=1)
    mask = np.array([True, False, True])
    rng = np.random.default_rng(1)
    assert all(pol.choose(np.zeros(2), mask, rng=rng)[0] != 1 for _ in range(200))


def test_after_a_turn_he_commits_to_the_new_way_for_a_while():
    w = Walk(100, 1, Navigator(seed=0), np.random.default_rng(0))
    w.want = 1
    w.since_turn = 0.0
    assert not w.mask()[BACK] and w.mask()[FORWARD]
    w.since_turn = navmod.COMMIT
    assert w.mask()[BACK]
    w.since_look = 0.0
    assert not w.mask()[LOOK]


def test_a_look_stands_still_and_turns_the_head_back_then_forward(qapp):
    from game.life import Life
    m = models.load()
    life = Life(m, 11, 7, 80, models.starter_brain(), seed=11)
    for _ in range(120):
        life.tick()
    while not (life.human.grounded and abs(life.human.vy) < 0.5):   # осматривается он только на земле
        life.tick()
    life.walk.look_left = navmod.LOOK_TIME
    x = life.human.px
    gazes = []
    for _ in range(int(navmod.LOOK_TIME * 60)):
        life.tick()
        gazes.append(life.human.gaze)
    assert abs(life.human.px - x) < 0.3
    assert -life.human.facing in gazes and gazes[0] == 0 and gazes[-1] == 0


def test_starter_navigator_was_trained():
    assert models.passport("navigator").get("training", {}).get("updates", 0) > 1000


@pytest.mark.parametrize("world,width", EXPLORED_WORLDS)
def test_in_an_explored_world_he_turns_by_himself_sometimes_but_does_not_dither(world, width):
    r = probe.wander(models.load(), models.starter_brain(), world, width, minutes=4)
    assert r["laps"] > 1.5, r
    assert r["voluntary_after"] > 0, r
    assert r["per_min_after"] <= MAX_PER_MIN and r["back_forth"] == 0, r


@pytest.mark.parametrize("world", [30, 31, 32])
def test_in_a_new_world_he_mostly_walks_on(world):
    """В основном вперёд: три четверти первого круга в одну сторону. Иногда он оборачивается -
    рассмотреть то, что только что прошёл (так и задумано), но не чаще раза в полминуты."""
    r = probe.wander(models.load(), models.starter_brain(), world, 320, minutes=1.5)
    assert r["first_lap_forward"] >= 0.75 and r["first_lap_turns"] <= 3, r


def test_a_detour_that_makes_progress_is_kept():
    """Застрял по дороге к цели - идёт в обход и не бросает обход, пока продвигается по нему
    (замер: обход через 8 с считался «застрял» по кратчайшему расстоянию, он ходил туда-сюда
    70 с у холма и пил на 100 с позже)."""
    w = Walk(160, 1, Navigator(seed=0), np.random.default_rng(0))
    w.goal = 30.0
    x, t = 150.0, 0.0
    for _ in range(int((navmod.STUCK + 1) * 10)):       # стоит у стены: ни на метр ближе
        w.tick(t, 0.1, x, 1, 1.0, can_decide=True)
        t += 0.1
    assert w.want == -1                                  # пошёл в обход
    for _ in range(300):                                 # 30 с идёт в обход по 1 м/с
        x -= 0.1
        w.tick(t, 0.1, x, -1, 1.0, can_decide=True)
        t += 0.1
        assert w.want == -1
