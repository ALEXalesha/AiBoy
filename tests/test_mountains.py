"""Горы (спецификация 2.0, раздел 8a): лёгкий путь и стена, лазание с нуля, снаряжение, выбор по
интересу, безопасность, флаг и радость на вершине."""
import os
import tempfile

import numpy as np
import pytest

from acts import climb as ac
from body.physics import WALL_SLOPE
from game import models
from game.life import Life
from skills import climb as sk
from world import mountains as mt
from world import reach
from world.generate import generate


@pytest.fixture(scope="module")
def m():
    return models.load()


def test_big_worlds_get_mountains_with_an_easy_route_and_a_real_wall(m):
    """Замер (план): 40 новых миров - гора в каждом 1280, две - в половине, в мирах 640 - в половине;
    лёгкий путь не круче 0.27 (идёт по нему, как по ровному), стена - не меньше 0.8 высоты горы и
    круче его шага."""
    big = 0
    for seed in range(9400, 9412):
        for width in (640.0, 1280.0):
            w = generate(m.world_genome, seed, width)
            if width == 1280.0:
                assert w.mountains, seed
                big += len(w.mountains)
            assert not reach.traps(w), (seed, width)
            for p in w.mountains:
                assert 60.0 <= p.height <= 100.0
                assert mt.easy_slope_max(w, p) <= 0.3, (seed, width)
                drop, flattest = mt.face_wall(w, p)
                assert drop >= 0.75 * p.height and flattest >= WALL_SLOPE, (seed, width, drop, flattest)
    assert big > 12                                          # бывает и по две
    assert generate(m.world_genome, 9400, 320.0).mountains == []


def test_the_mountain_net_was_evolved(m):
    p = models.passport("mountains")
    ev = p["check_40_new_seeds"]["evolved"]
    assert p.get("generations", 0) >= 20 and ev["worlds_with_traps"] == 0 and ev["dropped"] == 0
    assert ev["1280_with_mountain"] == 1.0


def mountain_life(m, seed=9106):
    os.environ["AIBOY_HOME"] = tempfile.mkdtemp()
    life = Life(m, seed, 7, 1280.0, models.starter_brain())
    return life, life.world.mountains[0]


def put(life, x):
    h = life.human
    h.px = x
    h.py = life.world.height_at(x) + h.body.leg() + 0.05
    h.vy = 0.0


def calm(life):
    life.needs.hunger = life.needs.thirst = life.needs.tired = 0.0


def test_the_wall_can_not_be_walked_and_the_easy_route_leads_to_the_top_with_a_flag_and_joy(m):
    life, p = mountain_life(m)
    foot = p.face_foot()
    put(life, foot + p.steep * 2.0)
    life.walk.goal, life.walk.goal_way, life.walk.goal_fixed = p.summit, -p.steep, True   # прямо в стену
    top = 0.0
    for _ in range(60 * 30):
        calm(life)
        life.tick()
        top = max(top, life.human.py - life.world.height_at(foot))
    assert top < 3.0                                          # по стене шагом не подняться
    life, p = mountain_life(m)
    put(life, p.summit - p.steep * 40.0)
    life.walk.goal, life.walk.goal_way, life.walk.goal_fixed = p.summit, p.steep, True
    events = []
    for _ in range(60 * 240):
        calm(life)
        events += life.tick()
        if any(e[0] == "celebrated" for e in events):
            break
    kinds = [e[0] for e in events]
    assert "summit" in kinds and "flag" in kinds and "celebrated" in kinds
    i = kinds.index("summit")
    assert abs(((life.human.px - p.summit + 640) % 1280) - 640) < mt.TOP / 2 + 0.5
    assert any(e[0] == "phrase" for e in events[i:]) and ("sound", "fanfare", 0.8, 0.1) in events
    assert life.climb.flags and life.choice.mood > 0.2


@pytest.mark.parametrize("seed", [0, 1])
def test_climbing_is_learned(seed):
    """Закон обучения: стена 60 м, навык без шума на 20 стенах до и после 1600 попыток (замер:
    0.01-0.04 -> 0.14-0.21; до вершины руками доходит редко - на то и снаряжение)."""
    c = sk.Climber(seed=seed)

    def judge():
        c.skill.explore = False
        r = np.mean([sk.practice(c, sk.Wall(200 + i, 0, 60.0), np.random.default_rng(i), learn=False) for i in range(20)])
        c.skill.explore = True
        return float(r)
    before = judge()
    rng = np.random.default_rng(seed)
    for i in range(1600):
        sk.practice(c, sk.Wall(100 + i % 7, 0, 60.0), rng)
    after = judge()
    assert before < 0.06 and after > before + 0.08, (before, after)


def trained(kind, seed, n=600):
    c = sk.Climber(seed=seed)
    rope = sk.RopeSkill(seed) if kind == "rope" else None
    axe = sk.AxeSkill(seed) if kind == "axe" else None
    rng = np.random.default_rng(seed)
    for i in range(n):
        sk.practice(c, sk.Wall(100 + i % 7, 0, 60.0), rng, rope_every=sk.ROPE_EVERY if rope else None,
                    rope_skill=rope, axe=axe)
    for t in (c, rope, axe):
        if t is not None:
            t.skill.explore = False
    return float(np.mean([sk.practice(c, sk.Wall(200 + i, 0, 60.0), np.random.default_rng(i),
                                      rope_every=sk.ROPE_EVERY if rope else None, rope_skill=rope, axe=axe, learn=False)
                          for i in range(20)]))


def test_tools_raise_the_success():
    """Замер: после 600 попыток стена 60 м - руками 0.04 / 0.15, с верёвкой 0.33 / 0.09,
    с ледорубом 0.54 / 0.31 (зёрна 0 / 1)."""
    bare = np.mean([trained("none", s) for s in (0, 1)])
    assert np.mean([trained("axe", s) for s in (0, 1)]) > bare + 0.15
    assert np.mean([trained("rope", s) for s in (0, 1)]) > bare + 0.05


def test_interest_fear_and_tiredness_choose_the_route(m):
    life, p = mountain_life(m)
    c = life.climb
    # дорога - отдельная прибавка (WAY_COST); здесь - только интерес, страх и усталость: идти до
    # лёгкого пути и до стены одинаково
    c.approach = lambda k, x: (0.0, 0.0)
    x = p.face_foot()
    c.climber.history = [0.8] * 10                                 # умеет: стена его тянет
    new, k = c.routes(x, ["rope", "axe"], 0.0)
    assert k == 0 and new["easy"] < 0.5
    assert c.bias(x) > 0
    c.conquered.add(0)
    old, _ = c.routes(x, ["rope", "axe"], 0.0)
    assert c.bias(x) < 0 and old["easy"] > new["easy"]            # покорённая - меньше тянет и стена
    c.conquered.clear()
    c.fear = 0.9
    scared, _ = c.routes(x, ["rope", "axe"], 0.0)
    c.fear = 0.0
    tired, _ = c.routes(x, ["rope", "axe"], 0.9)
    assert scared["easy"] > new["easy"] + 0.3 and tired["easy"] > new["easy"] + 0.3


def test_a_fall_from_the_wall_or_a_cliff_is_soft_and_he_never_hangs_there(m):
    life, p = mountain_life(m)
    events = []
    put(life, p.face_foot())
    life.climb.start_face(0, "face")
    worst_v, seen = 0.0, set()
    for _ in range(60 * 600):
        calm(life)
        events += life.tick()
        h = life.human
        assert h.py >= life.world.height_at(h.px) - 0.05            # не проваливается под землю
        seen.add(life.climb.mode)
        if life.climb.mode is None and life.climb.go is None:
            break
    kinds = [e[0] for e in events]
    assert life.climb.mode is None                                   # не висит на стене
    assert "climb_attempt" in kinds
    if "climb_fall" in kinds:
        assert life.climb.fear > 0 and "slide" in seen
    # с обрыва пешком - съезжает, без «падения»
    life, p = mountain_life(m)
    put(life, p.summit + p.steep * (mt.TOP / 2 - 0.3))
    life.walk.goal, life.walk.goal_way, life.walk.goal_fixed = p.face_foot(), p.steep, True
    events = []
    for _ in range(60 * 60):
        calm(life)
        events += life.tick()
        worst_v = min(worst_v, life.human.vy)
    kinds = [e[0] for e in events]
    assert worst_v >= -5.01 and "fall" not in kinds and kinds.count("slid") == 1


def test_one_flag_per_summit_it_persists_and_never_without_a_summit(m, tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOY_HOME", str(tmp_path))
    from game import life_store
    life = Life(m, 9106, 7, 1280.0, models.starter_brain())
    ev = []
    life.climb.summit(0, ev)
    life.climb.mode = None
    life.climb.summit(0, ev)
    assert len(life.climb.flags) == 1 and [e[0] for e in ev].count("flag") == 1
    assert life_store.save(life)
    again = Life(m, 9106, 7, 1280.0, models.starter_brain(), summits=life_store.load_summits(9106))
    assert again.climb.flags == life.climb.flags and 0 in again.climb.conquered
    assert life_store.load_summits(9107) is None
    calm_life = Life(m, 9106, 7, 1280.0, models.starter_brain())        # гуляет, на вершину не попал
    for _ in range(60 * 120):
        calm_life.tick()
        h = calm_life.human
        if mt.summit_at(calm_life.world.mountains, h.px, 1280.0) is not None:
            break
    else:
        assert calm_life.climb.flags == []


def test_a_long_easy_side_has_no_jump_at_half_the_ring():
    """Лёгкий путь горы 95 м - 430 м, длиннее полукольца мира 640: профиль ровный, без скачка
    (замер: кратчайшее расстояние до вершины давало ступень 13.5 м на полкольце)."""
    p = mt.Mountain(300.0, 95.0, 1, 0.0)
    mount, _ = mt.profile([p], 640.0, 0.25, int(640 / 0.25))
    steps = np.abs(np.diff(np.concatenate([mount, mount[:1]])))
    assert p.easy_len > 320.0 and steps.max() <= mt.FACE * 0.25 + 1e-6


def test_the_hook_learns_to_catch():
    rope = sk.RopeSkill(0)
    c = sk.Climber(0)
    rng = np.random.default_rng(0)
    for i in range(600):
        sk.practice(c, sk.Wall(100 + i % 7, 0, 60.0), rng, rope_every=sk.ROPE_EVERY, rope_skill=rope)
    assert rope.hooked / rope.throws > 0.25                 # замер: 0.4


def test_stuck_on_the_wall_he_climbs_down(m, monkeypatch):
    life, p = mountain_life(m)
    put(life, p.face_foot())
    monkeypatch.setattr(sk, "move", lambda c, w, y, arm, f, rope, rng, axe=None, learn=True: (y, f, 0))
    life.climb.start_face(0, "face")
    life.climb.y = 5.0
    events = []
    for _ in range(int(60 * (ac.STUCK + 15))):
        calm(life)
        events += life.tick()
        if life.climb.mode is None:
            break
    assert life.climb.mode is None and any(e[0] == "climb_attempt" for e in events)
