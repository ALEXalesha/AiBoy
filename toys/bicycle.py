"""Велосипед: рама и два колеса в pymunk (physics/world.py), шарниры на осях, мотор на заднем
колесе - это его ноги на педалях. Он решает, как сильно крутить педали и куда наклоняться
(вперёд в гору, назад с горы); голова - навык с гауссовым шумом, решение каждые STEP с по
тому, какой уклон впереди, как быстро едет и как наклонена рама. Учится REINFORCE по итогу
поездки (одна награда на все решения поездки, своя база).

Упал - рама перевернулась больше чем на FALL рад или поездка кончилась. Успех - сколько
проехал не упав (до GOOD м) и подъёмы. Новичок жмёт изо всех сил и не наклоняется - на
горке встаёт на дыбы. Звук: цепь (тихий тик по оборотам), падение - удар.
"""
import math

import numpy as np
import pymunk

from physics.world import PhysWorld
from skills.base import GaussianSkill

WHEEL_R = 0.3
BASE = 1.0               # м между осями
MASS = 12.0              # рама с ним
STEP = 0.25              # с между решениями
RIDE = 12.0              # с поездка
FALL = 1.0               # рад
GOOD = 40.0              # м - отличная поездка
MAX_RATE = 14.0          # рад/с колеса


class BikeHeads:
    def __init__(self, seed=0):
        # вход: уклон впереди, скорость/6, угол рамы, постоянная; выход: педали 0..1, наклон -1..1
        self.skill = GaussianSkill(4, [0.0, -1.0], [1.0, 1.0], hidden=16, seed=seed, sigma=0.25, lr=5e-3)
        unit = self.skill.bounds.to_unit(np.array([0.95, 0.0], np.float32))
        self.skill.net.items[-1].b[:] = np.arctanh(np.clip(unit, -0.95, 0.95))

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


class Bike:
    def __init__(self, phys, x, y, facing=1):
        self.phys = phys
        sp = phys.space
        self.facing = facing
        rear_x, front_x = x - facing * BASE / 2, x + facing * BASE / 2
        self.rear = pymunk.Body(1.5, pymunk.moment_for_circle(1.5, 0, WHEEL_R))
        self.front = pymunk.Body(1.5, pymunk.moment_for_circle(1.5, 0, WHEEL_R))
        self.rear.position, self.front.position = (rear_x, y), (front_x, y)
        self.frame = pymunk.Body(MASS, pymunk.moment_for_box(MASS, (BASE, 0.6)))
        self.frame.position = (x, y + 0.45)
        shapes = []
        for b in (self.rear, self.front):
            c = pymunk.Circle(b, WHEEL_R)
            c.friction, c.elasticity = 1.2, 0.1
            c.filter = pymunk.ShapeFilter(group=7)
            shapes.append(c)
        body_shape = pymunk.Poly.create_box(self.frame, (BASE * 0.8, 0.3))
        body_shape.friction = 0.6
        body_shape.filter = pymunk.ShapeFilter(group=7)
        shapes.append(body_shape)
        self.joints = [pymunk.PivotJoint(self.frame, self.rear, self.rear.position),
                       pymunk.PivotJoint(self.frame, self.front, self.front.position)]
        self.motor = pymunk.SimpleMotor(self.frame, self.rear, 0.0)
        self.motor.max_force = 40.0
        sp.add(self.rear, self.front, self.frame, *shapes, *self.joints, self.motor)
        self.items = [self.rear, self.front, self.frame, *shapes, *self.joints, self.motor]

    def remove(self):
        self.phys.space.remove(*self.items)

    def control(self, pedal, lean):
        # педали: колесо крутится вперёд (по взгляду); наклон - центр тяжести вперёд или назад
        self.motor.rate = self.facing * pedal * MAX_RATE
        self.frame.center_of_gravity = (self.facing * 0.25 * lean, 0.15)

    def angle(self):
        """Наклон рамы: плюс - задрал переднее колесо."""
        return self.facing * self.frame.angle

    @property
    def x(self):
        return float(self.frame.position.x)


class Flat:
    width, dx = 3000.0, 0.25

    def height_at(self, x):
        return 0.0


class Hills:
    """Холмы для замеров и тестов: подъёмы и спуски."""
    width, dx = 3000.0, 0.25

    def height_at(self, x):
        return 0.9 * math.sin(x / 6.0) + 0.5 * math.sin(x / 2.7 + 1.0)


class RideAttempt:
    name = "bicycle"

    def __init__(self, heads, ctx):
        self.heads, self.ctx = heads, ctx
        self.rng = ctx.rng if ctx.rng is not None else np.random.default_rng()
        self.phys = PhysWorld(ctx.world, center=ctx.x)
        g = ctx.ground(ctx.x)
        self.bike = Bike(self.phys, ctx.x, g + WHEEL_R + 0.1, ctx.facing)
        self.start = ctx.x
        self.t = 0.0
        self.clock = STEP
        self.records = []
        self.climb = 0.0
        self.last_y = float(self.bike.frame.position.y)
        self.trace = []
        self.cmd = (0.0, 0.0)

    def state(self):
        b = self.bike
        x = b.x
        f = b.facing
        slope = (self.ctx.ground(x + f * 1.5) - self.ctx.ground(x)) / 1.5
        v = f * float(b.frame.velocity.x)
        return np.array([np.clip(slope, -1.5, 1.5), v / 6.0, b.angle(), 1.0], np.float32)

    def step(self, dt, events):
        self.t += dt
        self.clock += dt
        if self.clock >= STEP:
            self.clock = 0.0
            cmd, rec = self.heads.skill.act(self.state())
            self.records.append(rec)
            self.cmd = (float(cmd[0]), float(cmd[1]))
            self.bike.control(*self.cmd)
            if self.cmd[0] > 0.2:
                events.append(("sound", "chain", 0.2 * self.cmd[0], 0.1))
        self.phys.follow(self.bike.x)
        self.phys.step(dt)
        y = float(self.bike.frame.position.y)
        self.climb += max(0.0, y - self.last_y)
        self.last_y = y
        self.trace.append((round(self.bike.x - self.start, 2), round(y, 2), round(self.bike.angle(), 2)))
        if abs(self.bike.angle()) > FALL:
            events.append(("sound", "fall", 0.9, 0.3))
            return self.finish(True)
        if self.t >= RIDE:
            return self.finish(False)
        return None

    def finish(self, fell):
        dist = self.bike.facing * (self.bike.x - self.start)
        success = float(np.clip(dist / GOOD, 0.0, 1.0)) * (0.5 if fell else 1.0) + 0.1 * min(1.0, self.climb / 5.0)
        success = min(1.0, success)
        self.heads.skill.learn_many(self.records, success)
        self.bike.remove()
        return success, {"distance": round(dist, 2), "fell": bool(fell), "climb": round(self.climb, 2),
                         "trace": self.trace[::6]}

    def finish_timeout(self):
        return self.finish(False)

    def pose(self, human):
        a = human.angles.copy()
        ph = self.t * 6.0 * max(0.2, self.cmd[0])
        a[0] = 0.35 + 0.25 * self.cmd[1]
        a[6] = 0.9 + 0.5 * math.sin(ph)
        a[7] = 0.9 - 0.5 * math.sin(ph)
        a[8] = 1.2 + 0.4 * math.cos(ph)
        a[9] = 1.2 - 0.4 * math.cos(ph)
        a[2] = a[3] = 1.1
        return a

    def visuals(self):
        b = self.bike
        return [("bike", b.x, float(b.frame.position.y),
                 (float(b.rear.position.x), float(b.rear.position.y), float(b.front.position.x),
                  float(b.front.position.y), float(b.rear.angle), float(b.frame.angle)), (0.2, 0.45, 0.85))]


def practice(heads, rng, world=None):
    from toys.inplace import Ctx, run
    return run(RideAttempt(heads, Ctx(world or Hills(), 0.0, rng=rng)), limit=RIDE + 1)[0]


from toys.inplace import InPlaceSession  # noqa: E402


class BikeSession(InPlaceSession):
    """Велосипед в жизни: он сидит на раме - тело едет вместе с велосипедом."""

    rides = True

    def attempt_tick(self, dt, events):
        res = super().attempt_tick(dt, events)
        self.place_body(final=res is not None)
        return res

    def place_body(self, final=False):
        life = self.life
        h = life.human
        a = self.attempt
        if a is None:
            return
        b = a.bike if not final else None
        if b is not None:
            h.px = b.x % life.world.width
            h.py = float(b.frame.position.y) + 0.12      # таз на седле
            h.tilt = -b.frame.angle * b.facing
            h.facing = b.facing
            h.angles = self.pose
            h.vx, h.vy = float(b.frame.velocity.x), 0.0
            h.grounded = False
        else:
            # поездка кончилась: слез (или упал) там, где был
            from body import skeleton
            x = a.trace[-1][0] + a.start if a.trace else h.px
            h.px = x % life.world.width
            h.tilt = 0.0
            rest = skeleton.rest_angles()
            h.angles, h.command, h.targets = rest.copy(), rest.copy(), rest.copy()
            h.py = life.world.height_at(h.px) + h.body.leg() + 0.05
            h.grounded = False

    def visuals(self):
        return self.attempt.visuals() if self.attempt is not None and self.attempt.t < RIDE else []
