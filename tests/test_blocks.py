"""Кубики: физика башни и учёба строить."""
import numpy as np

import toys.blocks as B
from physics.world import PhysWorld


def tower(offsets):
    p = PhysWorld(B.Flat(), center=0.0)
    y, x = 0.0, 0.0
    names = []
    for i, dx in enumerate(offsets):
        x += dx
        p.add_box(f"k{i}", x, y + 0.05 + 0.002, 0.2, 0.1)
        names.append((f"k{i}", x, y + 0.05))
        y += 0.1
        for _ in range(20):
            p.step(1 / 60)
    for _ in range(180):                                  # 3 с
        p.step(1 / 60)
    return max(abs(p.body(n).position.x - x0) + abs(p.body(n).position.y - y0) for n, x0, y0 in names)


def test_a_balanced_tower_stands_3_s_and_an_unbalanced_one_falls():
    assert tower([0.0] * 6) < 0.02
    assert tower([0.0, 0.08, 0.08, 0.08, 0.08]) > 0.1        # каждый сдвинут почти на полширины


def test_building_towers_is_learned_and_the_placing_head_learns_too():
    """Закон обучения: зерно 0, 300 башен; медиана последних 50 выше первых 50 (замер: 0.67 -> 0.83,
    зерно 1: 0.33 -> 0.83). И голова укладки сама: обученная без шума строит выше новичковой с
    той же головой выбора (замер: 1.0 против 0.5 и 0.42) - иначе учился бы только порядок кубиков."""
    import copy
    h = B.BlockHeads(seed=0)
    rng = np.random.default_rng(0)
    s = [B.practice(h, rng, seed=7) for _ in range(300)]
    assert np.median(s[-50:]) >= np.median(s[:50]) + 0.15
    newbie = copy.deepcopy(h)
    newbie.place = B.BlockHeads(seed=0).place
    for heads in (h, newbie):
        heads.place.explore = False

    def judge(heads):
        return np.median([B.practice(copy.deepcopy(heads), np.random.default_rng(100 + i), seed=7) for i in range(30)])
    assert judge(h) > judge(newbie) + 0.3


def test_block_sizes_come_from_the_toy_network_and_differ_by_chest():
    a, b = B.sizes(1), B.sizes(2)
    assert len(set(a)) == B.TOWER and a != b
    assert [w for w, _ in a] != [w for w, _ in b] and [h for _, h in a] != [h for _, h in b]


def test_a_tower_attempt_makes_sounds_and_a_journal_entry():
    from toys.inplace import Ctx
    att = B.TowerAttempt(B.BlockHeads(seed=3), Ctx(B.Flat(), 0.0, rng=np.random.default_rng(3), seed=7))
    events = []
    res = None
    for _ in range(60 * 30):
        res = att.step(1 / 60, events)
        if res is not None:
            break
    success, entry = res
    assert 0.0 <= success <= 1.0 and "blocks" in entry
    assert any(e[:2] == ("sound", "clack") for e in events)
    assert att.visuals() and att.visuals()[0][0] == "box"
