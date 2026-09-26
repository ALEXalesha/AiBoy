import math

import numpy as np
import pytest

from body import skeleton
from body.cppn import generate_body, random_genome
from body.physics import DT, Human


class Ground:
    """Мир для проверок: высота - любая функция x."""

    def __init__(self, fn, width=1000.0):
        self.fn, self.width = fn, width

    def height_at(self, x):
        return float(self.fn(x % self.width))


FLAT = Ground(lambda x: 0.0)


@pytest.fixture(scope="module")
def body():
    return generate_body(random_genome(np.random.default_rng(2)), 5)


def pose(**kw):
    a = skeleton.rest_angles()
    for name, v in kw.items():
        a[skeleton.JOINTS.index(name)] = v
    return a


GAIT = [pose(hip_l=0.8, kn_l=1.4, hip_r=-0.35, kn_r=0.05), pose(hip_l=0.35, kn_l=0.1, hip_r=-0.35, kn_r=0.05),
        pose(hip_r=0.8, kn_r=1.4, hip_l=-0.35, kn_l=0.05), pose(hip_r=0.35, kn_r=0.1, hip_l=-0.35, kn_l=0.05)]


def walk(h, seconds, turn=1.0):
    events = []
    n = int(seconds / DT)
    phase = int(0.3 / DT)
    for i in range(n):
        events += h.step(GAIT[(i // phase) % 4], turn=turn)
    return events


def settle(h, seconds=1.0):
    for _ in range(int(seconds / DT)):
        h.step(skeleton.rest_angles())


def feet_gap(h, world):
    return min(p[1] - world.height_at(p[0]) for p in (h.points()["ft_l"], h.points()["ft_r"]))


def test_standing_human_rests_on_the_ground(body):
    h = Human(body, FLAT, x=2.0)
    settle(h)
    assert h.grounded
    assert feet_gap(h, FLAT) == pytest.approx(0.0, abs=0.02)
    assert h.px == pytest.approx(2.0, abs=0.05)


def test_dropped_human_falls_down_to_the_ground(body):
    h = Human(body, FLAT, x=0.0)
    h.py += 1.0
    h.grounded = False
    h.contact = [False, False]
    settle(h, 1.5)
    assert h.grounded and feet_gap(h, FLAT) == pytest.approx(0.0, abs=0.02)


def test_pushing_the_stance_leg_back_moves_the_body_forward(body):
    h = Human(body, FLAT, x=0.0)
    settle(h)
    for _ in range(int(0.4 / DT)):
        h.step(pose(kn_l=1.6))                         # поднять левую ногу
    start = h.px
    assert abs(start) < 0.05
    for _ in range(int(0.6 / DT)):
        h.step(pose(hip_l=0.5, kn_l=1.6, hip_r=-0.4, kn_r=0.1))
    assert h.px - start > 0.2


def test_without_the_legs_nothing_moves_even_if_it_wants_to(body):
    h = Human(body, FLAT, x=0.0)
    settle(h)
    for _ in range(int(2 / DT)):
        h.step(skeleton.rest_angles(), turn=1.0)
    assert abs(h.px) < 0.05


def test_walking_gait_goes_where_the_human_looks(body):
    right = Human(body, FLAT, x=0.0)
    settle(right)
    walk(right, 6.0, turn=1.0)
    left = Human(body, FLAT, x=0.0)
    settle(left)
    walk(left, 6.0, turn=-1.0)
    assert right.px > 2.0 and left.px < -2.0
    assert right.odometer > 2.0


def test_jump_leaves_the_ground_and_lands(body):
    h = Human(body, FLAT, x=0.0)
    settle(h)
    y0 = h.py
    events = h.step(skeleton.rest_angles(), jump=True)
    assert "jump" in events and not h.grounded
    top, landed = y0, False
    for _ in range(int(1.5 / DT)):
        events = h.step(skeleton.rest_angles())
        top = max(top, h.py)
        landed = landed or "land" in events
    assert 0.5 < top - y0 < 1.4
    assert landed and h.grounded


def test_no_double_jump_in_the_air(body):
    h = Human(body, FLAT, x=0.0)
    settle(h)
    h.step(skeleton.rest_angles(), jump=True)
    for _ in range(5):
        assert "jump" not in h.step(skeleton.rest_angles(), jump=True)


def test_high_wall_stops_the_walk_and_a_low_ramp_does_not(body):
    wall = Ground(lambda x: float(np.clip((x - 3.0) / 0.25, 0, 1)) * 2.0)
    h = Human(body, wall, x=1.0)
    settle(h)
    walk(h, 8.0)
    assert 1.8 < h.px < 3.0
    flat = Human(body, FLAT, x=0.0)
    settle(flat)
    walk(flat, 8.0)
    assert flat.px > 3.5
    # ступенька ниже поднятой стопы и длинный подъём - проходятся той же походкой
    for fn in (lambda x: float(np.clip((x - 3.0) / 0.25, 0, 1)) * 0.12,
               lambda x: float(np.clip((x - 3.0) / 3.0, 0, 1)) * 1.0):
        h = Human(body, Ground(fn), x=1.0)
        settle(h)
        walk(h, 8.0)
        assert h.px > 0.75 * flat.px


def test_falling_from_a_cliff_knocks_the_human_down_and_he_gets_up(body):
    h = Human(body, FLAT, x=0.0)
    h.py += 6.0
    h.grounded = False
    h.contact = [False, False]
    events = []
    for _ in range(int(1.5 / DT)):
        events += h.step(skeleton.rest_angles())
    assert "fall" in events and h.fallen > 0
    assert abs(h.tilt) > 1.0
    for _ in range(int(2.0 / DT)):
        h.step(skeleton.rest_angles())
    assert h.fallen == 0 and h.tilt == 0
    assert feet_gap(h, FLAT) == pytest.approx(0.0, abs=0.03)


def test_brain_cannot_move_the_legs_while_lying(body):
    h = Human(body, FLAT, x=0.0)
    h.fallen = 1.0
    before = h.px
    for _ in range(20):
        h.step(pose(hip_r=-0.8, hip_l=1.5), jump=True)
    assert abs(h.px - before) < 0.05


def test_random_brain_never_breaks_physics(body):
    world = Ground(lambda x: 1.5 * math.sin(x / 3.0) + 0.4 * math.sin(x * 1.7))
    rng = np.random.default_rng(0)
    h = Human(body, world, x=0.0)
    for i in range(3000):
        if i % 10 == 0:
            targets = skeleton.random_angles(rng)
            turn = rng.uniform(-1, 1)
        h.step(targets, turn=turn, jump=bool(rng.random() < 0.02))
        assert np.isfinite([h.px, h.py, h.vx, h.vy]).all()
        pts = h.points()
        for name in ("ft_l", "ft_r"):
            x, y = pts[name]
            assert y >= world.height_at(x) - 0.05, (i, name)


def test_turning_on_the_spot_is_not_a_way_to_travel(body):
    """Замер 26.09: обученный мозг разворачивался 577 раз в минуту - разворот переставлял
    стопы, и туда-сюда с толчком ногой давало ход. Теперь при развороте стопы остаются на
    месте (бёдра зеркалятся), и дёрганье на месте никуда не ведёт."""
    h = Human(body, FLAT, x=0.0)
    settle(h)
    for i in range(int(6 / DT)):
        phase = (i // 12) % 2
        turn = 1.0 if phase == 0 else -1.0
        h.step(pose(kn_l=1.6, hip_r=-0.4 if phase == 0 else 0.4, kn_r=0.1), turn=turn)
    assert abs(h.px) < 1.0


def test_turns_are_not_more_often_than_the_pause(body):
    from body.physics import TURN_PAUSE
    h = Human(body, FLAT, x=0.0)
    settle(h)
    flips, last = [], h.facing
    for i in range(int(4 / DT)):
        h.step(skeleton.rest_angles(), turn=1.0 if (i // 21) % 2 == 0 else -1.0)
        if h.facing != last:
            flips.append(i * DT)
            last = h.facing
    assert len(flips) >= 3
    assert min(np.diff(flips)) >= TURN_PAUSE - 1e-9


def test_a_real_change_of_direction_still_turns_him(body):
    h = Human(body, FLAT, x=0.0)
    settle(h)
    for _ in range(int(0.5 / DT)):
        h.step(skeleton.rest_angles(), turn=-1.0)
    assert h.facing == -1
    for _ in range(int(0.5 / DT)):
        h.step(skeleton.rest_angles(), turn=1.0)
    assert h.facing == 1


def test_joints_start_and_stop_smoothly(body):
    """Цель сустава прыгнула на 1.5 рад - сустав трогается плавно, без рывка в первом кадре,
    и приходит без перелёта."""
    h = Human(body, FLAT, x=0.0)
    settle(h)
    j = skeleton.JOINTS.index("sh_r")
    start = h.angles[j]
    path = []
    for _ in range(int(1.5 / DT)):
        h.step(pose(sh_r=start + 1.5))
        path.append(h.angles[j])
    d = np.diff([start] + path)
    assert d[0] < 0.5 * d.max()                         # трогается, а не прыгает
    assert (d >= -1e-9).all()                           # без перелёта и возврата
    assert path[-1] == pytest.approx(start + 1.5, abs=0.02)


def test_a_short_flicker_of_the_wish_does_not_turn_him(body):
    """Разворот - только если желание держится TURN_HOLD: мелькнувшее «налево» на пару
    кадров (шум) взгляд не меняет."""
    from body.physics import TURN_HOLD
    h = Human(body, FLAT, x=0.0)
    settle(h)
    flicker = max(1, int(TURN_HOLD / DT) - 3)
    for _ in range(20):
        for _ in range(flicker):
            h.step(skeleton.rest_angles(), turn=-1.0)
        for _ in range(10):
            h.step(skeleton.rest_angles(), turn=1.0)
    assert h.facing == 1
    for _ in range(int(TURN_HOLD / DT) + 2):
        h.step(skeleton.rest_angles(), turn=-1.0)
    assert h.facing == -1


def test_body_keeps_its_own_clock(body):
    h = Human(body, FLAT, x=0.0)
    for _ in range(90):
        h.step(skeleton.rest_angles())
    assert h.clock == pytest.approx(1.5)
