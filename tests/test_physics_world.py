"""pymunk-мир предметов: рельеф, мяч, кубики, окно вокруг камеры, кисти и стопы."""
import math

import numpy as np
import pytest

from game import models
from physics import world as pw
from physics.world import PhysWorld
from world.generate import generate


@pytest.fixture(scope="module")
def bumpy():
    """Самый неровный из нескольких миров - там проваливаться проще всего."""
    m = models.load()
    worlds = [generate(m.world_genome, s, 160) for s in (3, 8, 13, 21, 34)]
    return max(worlds, key=lambda w: float(np.abs(np.diff(w.heights)).max()))


def above_ground(pw_, name, radius, slack=0.03):
    x, y = pw_.position(name)
    return y - radius >= pw_.ground(x) - SEG_TOL - slack


SEG_TOL = pw.SEG_R


@pytest.mark.parametrize("angle", [-60, -30, -10, 0, 10, 30, 60, 85])
def test_the_ball_never_falls_through_the_terrain_up_to_the_hardest_kick(bumpy, angle):
    for x0 in (10.0, 45.0, 90.0, 130.0):
        p = PhysWorld(bumpy, center=x0)
        r = 0.11
        p.add_ball("мяч", x0, bumpy.height_at(x0) + 0.3)
        a = math.radians(angle)
        p.body("мяч").velocity = (25.0 * math.cos(a), 25.0 * math.sin(a))
        for _ in range(60 * 5):
            p.follow(p.position("мяч")[0])
            p.step(1 / 60)
            assert above_ground(p, "мяч", r), (x0, angle, p.position("мяч"))


def test_a_box_comes_to_rest_on_flat_ground():
    class Flat:
        width, dx = 100.0, 0.25

        def height_at(self, x):
            return 0.0
    p = PhysWorld(Flat(), center=0.0)
    p.add_box("кубик", 0.0, 0.5, 0.2, 0.2, angle=0.3)
    for _ in range(60 * 3):
        p.step(1 / 60)
    b = p.body("кубик")
    assert b.position.y == pytest.approx(0.1 + pw.SEG_R, abs=0.02)
    assert math.hypot(*b.velocity) < 0.02 and abs(b.angular_velocity) < 0.05


def test_the_terrain_window_follows_and_a_rolling_ball_crosses_rebuilds_and_the_seam(bumpy):
    p = PhysWorld(bumpy, center=5.0)
    p.add_ball("мяч", 5.0, bumpy.height_at(5.0) + 0.3)
    b = p.body("мяч")
    rebuilt = 0
    for _ in range(60 * 60):
        b.velocity = (8.0, b.velocity.y)            # катится вправо, через шов кольца
        c = p.center
        p.follow(b.position.x)
        rebuilt += p.center != c
        p.step(1 / 60)
        assert above_ground(p, "мяч", 0.11)
    assert b.position.x > bumpy.width + 10 and rebuilt >= 10


def test_a_thing_left_far_behind_waits_where_it_was(bumpy):
    p = PhysWorld(bumpy, center=20.0)
    p.add_box("кубик", 20.0, bumpy.height_at(20.0) + 0.2, 0.2, 0.2)
    for _ in range(120):
        p.step(1 / 60)
    rest = p.position("кубик")
    p.follow(120.0)                                  # ушёл далеко
    assert "кубик" in p.frozen
    for _ in range(120):
        p.step(1 / 60)
    p.follow(25.0)                                   # вернулся
    assert "кубик" in p.things and p.position("кубик") == pytest.approx(rest, abs=1e-6)


def test_a_moving_foot_kicks_the_ball_with_its_own_speed():
    class Flat:
        width, dx = 100.0, 0.25

        def height_at(self, x):
            return 0.0
    p = PhysWorld(Flat(), center=0.0)
    p.add_ball("мяч", 0.0, 0.11 + pw.SEG_R + 0.001)
    foot = p.limb("стопа", 0.06)
    foot.position = (-0.5, 0.12)
    for i in range(20):
        p.move_limb("стопа", (-0.5 + 0.1 * (i + 1), 0.12), 1 / 60)   # 6 м/с вправо
        p.step(1 / 60)
    vx = p.body("мяч").velocity.x
    assert 4.0 < vx < 14.0
