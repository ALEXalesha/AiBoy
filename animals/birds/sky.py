"""Небо мира: когда прилетают стайки и сколько в них птиц (спецификация 2.0, разделы 9 и 10).

Настройка «Птицы»: нет / редко / часто - основа (шанс в минуту). Настроение человечка её меняет:
в хорошем стайки прилетают чаще и больше (до 8), в грустном - реже и по 1-2. «Нет» - всегда нет.
"""
import numpy as np

from animals.birds import flight as fl
from animals.birds.flock import LEAVE, Flock, TakeOff
from animals.birds.look import look as bird_look

MODES = ("off", "rare", "often")
MODE_NAMES = {"off": "Нет", "rare": "Редко", "often": "Часто"}
RATE = {"off": 0.0, "rare": 0.15, "often": 0.45}     # стаек в минуту при ровном настроении
MAX_FLOCKS = 3


def rate(mode, mood):
    m = max(-1.0, min(1.0, float(mood)))
    return RATE.get(mode, 0.0) * max(0.2, min(1.8, 1.0 + 0.8 * m))


def flock_size(mood, rng):
    """Грустный - 1-2; иначе 2-8, тем больше, чем веселее."""
    m = max(-1.0, min(1.0, float(mood)))
    if m < -0.3:
        return int(rng.integers(1, 3))
    top = 2 + int(round(6 * (m + 1.0) / 2.0))
    return int(rng.integers(2, max(3, top + 1)))


class Sky:
    def __init__(self, world, world_seed, mode="off", rng=None, flyer=None, takeoff=None):
        self.world = world
        self.mode = mode
        self.rng = rng or np.random.default_rng(world_seed)
        self.look = bird_look(world_seed)
        self.flyer = flyer or fl.Flyer(seed=int(world_seed) % (2 ** 31))
        self.takeoff = takeoff or TakeOff(seed=int(world_seed) % (2 ** 31) + 1)
        self.flocks = []
        self.clock = 0.0
        self.seen = 0

    def spawn(self, x_near, mood):
        side = 1 if self.rng.random() < 0.5 else -1
        n = flock_size(mood, self.rng)
        f = Flock(self.world, self.look, self.flyer, self.takeoff, x_near - side * 25.0, side, n, self.rng)
        self.flocks.append(f)
        self.seen += n
        return f

    def tick(self, dt, x_near, mood, dangers, wind=0.0, learn=True):
        """Шаг неба. Возвращает события (звуки, бегство)."""
        events = []
        self.clock += dt
        if self.clock >= 1.0:
            self.clock = 0.0
            if len(self.flocks) < MAX_FLOCKS and self.rng.random() < rate(self.mode, mood) / 60.0:
                self.spawn(x_near, mood)
                events.append(("birds_came", len(self.flocks[-1].birds)))
        for f in list(self.flocks):
            f.tick(dt, dangers, wind, learn)
            events += f.events
            f.events = []
            d = abs((f.x - x_near + self.world.width / 2) % self.world.width - self.world.width / 2)
            if f.state == "leave" and d > LEAVE:
                self.flocks.remove(f)
        return events

    def perched_x(self):
        return [b.x for f in self.flocks for b in f.perched()]

    def visuals(self):
        return {"look": self.look, "birds": [v for f in self.flocks for v in f.visuals()]}
