"""Собака без жизни вокруг - для законов и замеров: он подходит (терпеливо или бегом), бросает
мяч, собака гуляет одна по миру. Всё - тем же кодом, что в игре (Dog.tick), только «он» -
простая точка с заданной скоростью.
"""
import math

import numpy as np

from animals.dog.dog import Dog, DogCtx, ring
from physics.world import PhysWorld


class Flat:
    width, dx = 200.0, 0.25

    def height_at(self, x):
        return 0.0


def _step(dog, phys, ctx, dt=1 / 60):
    ctx.events = []
    phys.follow(dog.x)
    dog.tick(dt, ctx)
    phys.step(dt)
    return ctx.events


def approach(world_seed, patient, minutes=4.0, trust=0.0, speed=None):
    """Он подходит к собаке: терпеливо (0.5 м/с, останавливается у её границы, протягивает
    лакомство раз в 8 с) или бегом (speed, по умолчанию 1.5 м/с, прямо на неё).
    Возвращает: доверие по полминуты, убегала ли, среднюю скорость собаки от него, ручная ли."""
    w = Flat()
    phys = PhysWorld(w, 50.0)
    dog = Dog(world_seed, phys, 50.0)
    dog.trust = trust
    ctx = DogCtx(w, phys)
    px, offer, fled, away, log = 35.0, 0.0, 0, [], []
    speed = speed or (0.5 if patient else 1.5)
    for k in range(int(minutes * 3600)):
        d = ring(px, dog.x, w.width)
        if patient:
            v = math.copysign(speed, d) if abs(d) > dog.keep_away() + 0.8 else 0.0
            offer += 1 / 60
            if abs(d) < 3.0 and offer > 8.0:
                offer = 0.0
                dog.offered(ctx)
        else:
            v = math.copysign(speed, d) if abs(d) > 1.0 else 0.0
        px += v / 60
        ctx.person = (px, v)
        was = dog.fleeing
        x0 = dog.x
        _step(dog, phys, ctx)
        if dog.fleeing > 0 and was == 0:
            fled += 1
        if dog.fleeing > 0:
            away.append(math.copysign(1.0, d) * (dog.x - x0) * 60)
        if k % 1800 == 0:
            log.append(round(dog.trust, 3))
    log.append(round(dog.trust, 3))
    return {"trust": log, "fled": fled, "away_speed": float(np.mean(away)) if away else 0.0, "tamed": dog.tamed,
            "dog": dog}


def fetch(world_seed, throws=60, seed=None, dog=None, praise=True, needs=True):
    """Ручная собака, он стоит и бросает мяч на 5-10 м; принесла - хвалит и даёт лакомство;
    мяч остался далеко (убежала с ним, бросила) - через 25 с бросает снова. Возвращает решения
    собаки при каждой хватке, вероятность «принести» её сети до и после, смены занятий.
    praise=False - не хвалит (закон: без похвалы нести не учится); needs=False - нужды на нуле
    (закон: бросает мяч от скуки, а не от голода)."""
    w = Flat()
    phys = PhysWorld(w, 50.0)
    if dog is None:
        dog = Dog(world_seed, phys, 52.0)
    dog.trust, dog.tamed = 0.9, True
    ctx = DogCtx(w, phys)
    px = 50.0
    ctx.person = (px, 0.0)
    rng = np.random.default_rng(world_seed if seed is None else seed)
    x = np.array([dog.trust, dog.look.playful, 1.0])
    before = float(dog.fetch.probs(x)[0])
    grabs, n, wait, k = [], 0, 0.0, 0
    switches = []
    prev = dog.choice.current

    def throw():
        phys.remove("мяч")
        phys.add_ball("мяч", px + 0.3, 1.5, radius=0.11)
        d = rng.uniform(5.0, 10.0)
        v = math.sqrt(d * 9.81 / math.sin(1.4))
        phys.body("мяч").velocity = (v * math.cos(0.7), v * math.sin(0.7))
        ctx.thrown = True

    throw()
    n = 1
    while n <= throws and k < 60 * 60 * 40:
        k += 1
        b = phys.body("мяч")
        ctx.ball = (float(b.position.x), float(b.position.y), float(b.velocity.length)) if b is not None else None
        for e in _step(dog, phys, ctx):
            if e[0] == "dog_grab":
                grabs.append(e[1])
            elif e[0] == "dog_brought" and praise:
                dog.praised(True)
        if not needs:
            dog.hunger = dog.thirst = dog.tired = 0.0
        if dog.choice.current != prev:
            switches.append((prev, dog.choice.current, round(k / 60, 1)))
            prev = dog.choice.current
        wait += 1 / 60
        b = phys.body("мяч")
        if b is not None and not dog.carrying and (abs(b.position.x - px) < 1.5 and b.velocity.length < 0.5 or wait > 25.0):
            throw()
            n += 1
            wait = 0.0
    after = float(dog.fetch.probs(np.array([dog.trust, dog.look.playful, 1.0]))[0])
    return {"grabs": grabs, "before": before, "after": after, "brought": dog.brought, "switches": switches,
            "minutes": k / 3600, "dog": dog}


def wander(world_seed, world, minutes=3.0, x0=None, water=None, dog_seed=None):
    """Собака одна в мире. Возвращает самый долгий отрезок (с), когда она хотела идти, а за
    10 с не прошла и метра (путь, а не смещение: сходила и вернулась - не застряла), и сколько
    прошла всего."""
    phys = PhysWorld(world, x0 or world.width * 0.3)
    dog = Dog(world_seed if dog_seed is None else dog_seed, phys, x0 or world.width * 0.3)
    ctx = DogCtx(world, phys, water)
    hist, worst, walked, last = [], 0.0, 0.0, dog.x
    stuck = 0.0
    for k in range(int(minutes * 3600)):
        _step(dog, phys, ctx)
        walked += abs(ring(last, dog.x, world.width))
        last = dog.x
        if k % 60 == 0:
            hist.append((dog.x, dog.want > 0))
            if len(hist) > 10:
                hist.pop(0)
                moving = all(h[1] for h in hist)
                moved = sum(abs(ring(a[0], b[0], world.width)) for a, b in zip(hist, hist[1:]))
                if moving and moved < 1.0:
                    stuck += 1.0
                    worst = max(worst, stuck + 9.0)
                else:
                    stuck = 0.0
    return {"worst_stuck": worst, "walked": walked, "dog": dog}
