"""Игрушки, с которыми играют на месте (кубики, пузыри, самолётик, бумеранг, удочка,
фломастер): одна и та же попытка и без жизни вокруг (замеры, законы обучения), и в жизни
(шаг за шагом, с рисованием и звуком).

У игрушки - класс попытки: Attempt(heads, ctx).step(dt, events) возвращает (успех, запись
журнала) или None, пока идёт; visuals() - что рисовать. ctx - где он и что вокруг: мир,
ветер, вода, x, взгляд, рост, сила рук. InPlaceSession - сессия в жизни: попытки подряд, он
стоит (или сидит), поза - от попытки.
"""
from dataclasses import dataclass, field

from toys.base import Session


@dataclass
class Ctx:
    world: object
    x: float
    facing: int = 1
    height: float = 1.7            # рост, м
    arm: float = 0.55              # рука (плечо + предплечье), м
    wind: object = None            # world/weather.Wind или None
    water: object = None           # world/water.Water или None
    fish: object = None
    t: float = 0.0
    rng: object = None
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def ground(self, x):
        return self.world.height_at(x)

    def wind_at(self, x, y, t):
        return self.wind.at(x, y, t) if self.wind is not None else 0.0


def run(attempt, dt=1 / 60, limit=60.0):
    """Попытка без жизни вокруг - до конца."""
    t = 0.0
    events = []
    while t < limit:
        res = attempt.step(dt, events)
        if res is not None:
            return res
        t += dt
    return attempt.finish_timeout()


class InPlaceSession(Session):
    """Сессия в жизни: попытки подряд, на месте."""

    def __init__(self, life, name, heads, journal_, make_attempt):
        super().__init__(life, heads, journal_)
        self.name = name
        self.make_attempt = make_attempt
        self.attempt = None
        self.pose = None

    def ctx(self):
        life = self.life
        h = life.human
        return Ctx(life.world, h.px, h.facing, h.body.height(), h.body.upper_arm + h.body.forearm,
                   getattr(life, "wind", None), life.water, life.fish, life.time, life.rng, life.body_seed)

    def start_attempt(self):
        self.attempt = self.make_attempt(self.head, self.ctx())
        self.pose = self.attempt.pose(self.life.human)

    def attempt_tick(self, dt, events):
        self.attempt.ctx.t = self.life.time
        res = self.attempt.step(dt, events)
        self.pose = self.attempt.pose(self.life.human)
        return res

    def visuals(self):
        return self.attempt.visuals() if self.attempt is not None else []
