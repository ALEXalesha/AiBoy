"""Стайки птиц (спецификация 2.0, раздел 9).

Стайка прилетает по небу на своей высоте, садится на землю впереди, клюёт; кто-то (он, собака)
ближе «расстояния взлёта» - взлетает и улетает дальше (иногда садится снова, потом улетает
совсем). Высоту держит сеть полёта породы (animals/birds/flight.py), она учится и здесь.

**Когда взлетать** - своя маленькая голова (GaussianSkill) на породу: по тому, как быстро к ним
приближаются, решает расстояние взлёта. Взлетели заранее и далеко - хорошо, но кормились меньше
(цена за метр); подпустили слишком близко - испуг (минус). Так они учатся взлетать раньше, если
их пугали.

**Закон мира:** никто не ловит птицу. Ближе CATCH к сидящей птице подойти нельзя - она взлетает
сразу, что бы ни решила голова.
"""
import math

import numpy as np

from animals.birds import flight as fl
from skills.base import GaussianSkill

CATCH = 0.6                 # м - ближе не подойти никому
TAKEOFF = (2.0, 10.0)       # м - расстояние взлёта
SCARED = 0.5                # м - ближе - испугались (минус голове взлёта)
SCARE_TIME = 1.0            # с до встречи - тоже испуг
EARLY_COST = 0.05           # за метр раннего взлёта
CRUISE_H = (2.5, 4.2)       # м над землёй (в кадре над ним видно около 5 м)
PERCH = (8.0, 25.0)         # с клюют на земле
LEAVE = 60.0                # м от него - улетели совсем


class TakeOff:
    def __init__(self, seed=0):
        self.skill = GaussianSkill(2, [TAKEOFF[0]], [TAKEOFF[1]], hidden=8, seed=seed, sigma=0.3, lr=1e-2)
        # новички подпускают близко (около 2.2 м) - пугаются и учатся взлетать раньше
        self.skill.net.items[-1].b[:] = np.float32(np.arctanh(-0.9))
        self.scares = 0
        self.flights = 0

    def decide(self, approach, glider):
        cmd, rec = self.skill.act(np.array([min(1.0, approach / 3.0), 1.0 if glider else 0.0], np.float32))
        return float(cmd[0]), rec

    def learn(self, rec, closest, distance, speed=0.0, learn=True):
        """Испуг - подпустили ближе, чем за SCARE_TIME до встречи (и не меньше SCARED): к бегущему
        надо взлетать раньше (замер: с порогом 1.5 м испуга не было никогда - учиться не по чему)."""
        too_close = closest < SCARED + speed * SCARE_TIME
        self.flights += 1
        self.scares += int(too_close)
        if learn and rec is not None:
            self.skill.learn(rec, (-1.0 if too_close else 0.5) - EARLY_COST * distance)

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


class Flock:
    def __init__(self, world, look, flyer, takeoff, x, facing, n, rng):
        self.world, self.look, self.flyer, self.takeoff, self.rng = world, look, flyer, takeoff, rng
        self.wings = fl.Wings(look)
        self.facing = facing
        self.height = float(rng.uniform(*CRUISE_H))
        self.birds = []
        for k in range(n):
            bx = x - facing * k * 0.8 + float(rng.normal(0, 0.3))
            b = fl.Bird(bx, world.height_at(bx) + self.height + float(rng.normal(0, 0.5)), self.wings.cruise, self.wings, facing)
            b.state = "fly"
            self.birds.append(b)
        self.state = "fly"          # "fly" | "land" | "perch" | "flee" | "leave"
        self.land_x = None
        self.perch_t = 0.0
        self.landings = 0
        self.decide_t = 0.0
        self.recs = {}
        self.distance = None        # расстояние взлёта этой посадки
        self.speed = 0.0
        self.rec = None
        self.land_t = 0.0
        self.closest = math.inf
        self.gone = False
        self.events = []

    @property
    def x(self):
        return float(np.mean([b.x for b in self.birds]))

    def perched(self):
        return [b for b in self.birds if b.state == "perch"]

    def plan_landing(self):
        self.land_x = self.x + self.facing * float(self.rng.uniform(15.0, 30.0))
        self.state = "land"

    def tick(self, dt, dangers, wind=0.0, learn=True):
        """dangers - [(x, скорость к птицам)] - он и собака."""
        w = self.world
        self.decide_t += dt
        decide = self.decide_t >= fl.DECIDE
        if decide:
            self.decide_t = 0.0
        if self.state == "perch":
            self.perch_t -= dt
            near = min((abs(self._ring(d[0] - self.x)) for d in dangers), default=math.inf)
            self.closest = min(self.closest, near)
            speed = max((d[1] for d in dangers), default=0.0)
            if self.distance is None or speed > self.speed + 0.5:
                self.speed = speed
                self.distance, self.rec = self.takeoff.decide(speed, self.look.glider)
            forced = any(abs(self._ring(d[0] - b.x)) < CATCH for d in dangers for b in self.birds)
            if near < self.distance or forced or self.perch_t <= 0:
                self.takeoff.learn(self.rec if near < self.distance or forced else None, self.closest, self.distance,
                                   self.speed, learn)
                if dangers and (near < self.distance or forced):
                    nearest = min(dangers, key=lambda d: abs(self._ring(d[0] - self.x)))
                    self.facing = 1 if self._ring(self.x - nearest[0]) >= 0 else -1   # от него
                    self.events.append(("birds_flee", len(self.birds)))
                for b in self.birds:
                    b.state, b.facing, b.vx, b.vy = "fly", self.facing, self.wings.cruise * 0.6, 2.5
                self.events.append(("sound", "flap", 0.6, self.look.span))
                self.state = "flee" if self.landings < 2 and self.rng.random() < 0.5 else "leave"
                self.distance, self.closest = None, math.inf
            else:
                for b in self.birds:                          # клюют, переминаются
                    b.phase = (b.phase + dt * 0.5) % 1.0
                    b.y = w.height_at(b.x)
            return
        if self.state == "land" and self.land_x is not None and any(abs(self._ring(d[0] - self.land_x)) < 3.0 + d[1]
                                                                      for d in dangers):
            self.land_x += self.facing * 15.0          # там кто-то - сядут дальше
        for b in self.birds:
            ground = w.height_at(b.x)
            if b.state == "perch":
                if any(abs(self._ring(d[0] - b.x)) < CATCH for d in dangers):
                    b.state, b.vx, b.vy = "fly", self.wings.cruise * 0.6, 2.5   # никто не ловит птицу
                    self.state = "flee"
                    self.events.append(("sound", "flap", 0.6, self.look.span))
                continue
            target = ground + self.height
            if self.state == "land" and self.land_x is not None and self._ring(self.land_x - b.x) * self.facing < 8.0:
                target = ground                               # садится (и пролетев место - тоже, иначе не садилась)
            if decide:
                feats = self.flyer.features(b, b.y, target, wind)
                cmd, rec = self.flyer.skill.act(feats)
                b.cmd = tuple(float(c) for c in cmd)
                prev = self.recs.get(id(b))
                if prev is not None and learn:
                    pr, ph = prev
                    self.flyer.skill.learn(pr, fl.Flyer.reward(b.y - ph, (b.vx - b.w.cruise) / b.w.cruise, b.cmd[0]))
                self.recs[id(b)] = (rec, target)
            b.facing = self.facing
            b.step(dt, wind)
            if b.y < ground + 0.15:
                b.y = ground
                near = any(abs(self._ring(d[0] - b.x)) < TAKEOFF[0] for d in dangers)
                if self.state == "land" and not near:
                    b.state, b.vx, b.vy = "perch", 0.0, 0.0
                elif near:
                    b.vy = max(b.vy, 2.5)                     # рядом кто-то - не садится (закон: не поймать)
                else:
                    b.vy = max(b.vy, 2.0)                     # чуть не села - снова вверх
        down = sum(b.state == "perch" for b in self.birds)
        self.land_t = self.land_t + dt if self.state == "land" and down else 0.0
        if self.state == "land" and (down == len(self.birds) or (down * 2 >= len(self.birds) and self.land_t > 4.0)):
            for b in self.birds:                              # отставшие садятся рядом со своими
                if b.state != "perch":
                    b.state, b.vx, b.vy, b.y = "perch", 0.0, 0.0, w.height_at(b.x)
            self.state, self.perch_t = "perch", float(self.rng.uniform(*PERCH))
            self.landings += 1
            self.events.append(("sound", "chirp", 0.5, self.look.span))
        elif self.state in ("fly", "flee") and self.rng.random() < dt / 6.0:
            self.plan_landing()

    def _ring(self, d):
        wd = self.world.width
        return (d + wd / 2) % wd - wd / 2

    def visuals(self):
        return [(b.x, b.y, b.phase, b.facing, b.state == "perch") for b in self.birds]
