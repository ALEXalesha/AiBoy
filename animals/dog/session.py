"""Его занятие «собака» (спецификация 2.0, разделы 8 и 10): приручать неручную, играть с ручной.

**Приручать** - голова учится (skills/base.GaussianSkill): по доверию собаки (видно по тому,
как она держится), расстоянию и тому, убегала ли она, решает, как быстро подходить (доля
времени, когда он идёт, - остальное стоит), где остановиться и с какого расстояния протянуть
лакомство из рюкзака. Успех попытки - насколько выросло доверие. Подходит быстро (он ходит около
2 м/с) - собака видит, что на неё бегут, и убегает: голова учится не спешить.

**Играть** с ручной: мяч в руках - бросает, собака бежит за ним; принесла - похвала и лакомство
(это делает жизнь, Life), успех 1; убежала с мячом или бросила - успех меньше, и он сам идёт за
мячом. Мяча нет - гладит её: подходит и стоит рядом.
"""
import math

import numpy as np

from skills.base import GaussianSkill
from toys.base import Session

CYCLE = 1.2                 # с - шаг-пауза: идёт долю duty, потом стоит
TAME_MAX = 30.0             # с - попытка приручить
WAIT_MAX = 6.0              # с - ждёт на месте
THROW_WAIT = 25.0           # с - ждёт, принесёт ли
WIND_UP = 0.35
REACH = 0.7                 # м - дотянулся до мяча
PET = 3.0                   # с гладит


class TameHeads:
    def __init__(self, seed=0):
        # вход: доверие, расстояние/10, убегала ли; выход: доля ходьбы, где встать, откуда протянуть
        self.skill = GaussianSkill(3, [0.1, 0.8, 0.6], [1.0, 5.0, 2.5], hidden=12, seed=seed, sigma=0.25, lr=5e-3)

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


class DogSession(Session):
    name = "dog"

    def __init__(self, life, heads, journal_):
        super().__init__(life, heads, journal_)
        self.kind = None
        self.stage = None
        self.t = 0.0
        self.pose = None
        self.cmd = None
        self.record = None
        self.trust0 = 0.0
        self.offered = False
        self.took = False
        self.outcome = None

    @property
    def dog(self):
        return self.life.dog

    def _gap(self):
        life = self.life
        return (self.dog.x - life.human.px + life.world.width / 2) % life.world.width - life.world.width / 2

    def _stand(self):
        from game.life import STAND
        a = self.life.human.angles.copy()
        a[2:6] = STAND[2:6]
        return a

    # --- попытки ---
    def start_attempt(self):
        life = self.life
        dog = self.dog
        self.t, self.offered, self.took, self.outcome = 0.0, False, False, None
        if dog is None:
            self.kind = None
            return
        if not dog.tamed:
            self.kind, self.stage = "tame", "approach"
            feats = np.array([dog.trust, min(1.0, abs(self._gap()) / 10.0), 1.0 if dog.fleeing > 0 else 0.0], np.float32)
            cmd, self.record = self.head.skill.act(feats)
            self.cmd = tuple(float(c) for c in cmd)
            self.trust0 = dog.trust
        elif life.inv.where.get("ball") == "carried":
            self.kind, self.stage = "throw", "windup"
        else:
            self.kind, self.stage = "pet", "approach"
        self.pose = None

    def attempt_tick(self, dt, events):
        life = self.life
        dog = self.dog
        if dog is None or self.kind is None:
            return 0.0, {"kind": "gone"}
        self.t += dt
        gap = self._gap()
        if self.kind == "tame":
            return self._tame(dt, gap, events)
        if self.kind == "throw":
            return self._throw(dt, gap, events)
        # гладит
        if abs(gap) > 1.0 and self.t < 20.0:
            life.walk.goal = dog.x
            self.pose = None
            return None
        life.walk.goal = None
        self.pose = self._stand()
        if self.stage != "pet":
            self.stage, self.t = "pet", 0.0
        if self.t >= PET:
            near = abs(gap) < 1.6
            if near:
                dog.trust = min(1.0, dog.trust + 0.01)
            return (1.0 if near else 0.3), {"kind": "pet", "near": bool(near)}
        return None

    def _tame(self, dt, gap, events):
        life = self.life
        dog = self.dog
        duty, stop, offer = self.cmd
        if self.stage == "approach":
            if abs(gap) <= stop or self.t > TAME_MAX:
                self.stage, self.t = "wait", 0.0
            else:
                walking = (self.t % CYCLE) < duty * CYCLE
                life.walk.goal = dog.x if walking else None
                self.pose = None if walking else self._stand()
                return None
        life.walk.goal = None
        self.pose = self._stand()
        if not self.offered and abs(gap) <= offer and life.backpack.treats > 0:
            self.offered = True
            self.took = dog.offered(life.dog_ctx)
            if self.took:
                life.backpack.treats -= 1
            events.append(("sound", "treat", 0.4, 0.05))
        if self.offered or self.t > WAIT_MAX:
            success = float(np.clip(0.5 + 6.0 * (dog.trust - self.trust0), 0.0, 1.0))
            self.head.skill.learn(self.record, success)
            return success, {"kind": "tame", "trust": round(dog.trust, 3), "approach": round(duty, 2),
                             "stop": round(stop, 2), "offer": round(offer, 2), "offered": self.offered,
                             "took": bool(self.took), "tamed": bool(dog.tamed)}
        return None

    def _throw(self, dt, gap, events):
        life = self.life
        h = life.human
        if self.stage == "windup":
            a = h.angles.copy()
            k = min(1.0, self.t / WIND_UP)
            a[3], a[5] = 2.6 - 1.4 * k, 1.0 * (1 - k)
            self.pose = a
            if self.t >= WIND_UP:
                self._release(events)
            return None
        self.pose = self._stand()
        if self.outcome is not None or self.t > THROW_WAIT:
            outcome = self.outcome or "ignored"
            success = {"brought": 1.0, "ran_off": 0.3, "ignored": 0.1}.get(outcome, 0.1)
            self.stage = "get_ball"
            return success, {"kind": "throw", "outcome": outcome}
        return None

    def _release(self, events):
        life = self.life
        h = life.human
        d = float(life.rng.uniform(5.0, 10.0))
        angle = 0.7
        v = math.sqrt(d * 9.81 / math.sin(2 * angle))
        x0 = h.px + h.facing * 0.35
        y0 = max(life.world.height_at(x0) + 0.2, h.py + 0.5)
        phys = life.physics()
        phys.remove("мяч")
        phys.add_ball("мяч", x0, y0, radius=0.11, mass=0.4)
        phys.body("мяч").velocity = (h.facing * v * math.cos(angle), v * math.sin(angle))
        life.inv.where["ball"] = "flying"
        life.dog_ctx.thrown = True
        events.append(("sound", "throw", min(1.0, v / 15.0), 0.11))
        self.stage, self.t = "wait", 0.0

    def dog_event(self, kind):
        """Собака принесла / убежала с мячом / бросила - от жизни."""
        if self.kind == "throw" and self.stage == "wait" and self.outcome is None:
            self.outcome = {"dog_brought": "brought", "dog_ran_off": "ran_off"}.get(kind)

    def ready(self):
        return self.kind != "throw" or self.life.inv.where.get("ball") == "carried"

    def tick(self, dt, events):
        """Между бросками он идёт за мячом (если собака не принесла) и подбирает его."""
        if not self.active:
            return False
        if self.kind == "throw" and self.stage == "get_ball":
            life = self.life
            b = life.physics().body("мяч")
            x = float(b.position.x) if b is not None else life.inv.where.get("ball")
            if isinstance(x, float):
                life.inv.where["ball"] = x
                if life.inv.dist(life.human.px, x) < REACH:
                    life.physics().remove("мяч")
                    life.inv.where["ball"] = "carried"
                    life.walk.goal = None
                    self.pose = None
                    self.start_attempt()
                    return True
                life.walk.goal = x
                self.pose = None
            else:
                # мяч у собаки в зубах - ждёт
                life.walk.goal = None
                self.pose = self._stand()
            return True
        return super().tick(dt, events)


def tame_practice(heads, world_seed, rng, walk=2.0):
    """Одна попытка приручить без жизни вокруг (законы, замеры): новая неручная собака того же
    мира с доверием 0-0.4, он в 6-12 м, идёт со скоростью walk долю duty каждого шага-паузы,
    встаёт, ждёт до WAIT_MAX, протягивает лакомство. Успех - как в игре: 0.5 + 6 x рост доверия."""
    from animals.dog.dog import Dog, DogCtx, ring
    from animals.dog.practice import Flat
    from physics.world import PhysWorld
    w = Flat()
    phys = PhysWorld(w, 60.0)
    dog = Dog(world_seed, phys, 60.0, rng=np.random.default_rng(rng.integers(2 ** 31)))
    dog.trust = float(rng.uniform(0.0, 0.4))
    ctx = DogCtx(w, phys)
    px = 60.0 - float(rng.uniform(6.0, 12.0)) * (1 if rng.random() < 0.5 else -1)
    feats = np.array([dog.trust, min(1.0, abs(ring(px, dog.x, w.width)) / 10.0), 0.0], np.float32)
    cmd, rec = heads.skill.act(feats)
    duty, stop, offer = (float(c) for c in cmd)
    trust0, t, stage, offered = dog.trust, 0.0, "approach", False
    while t < TAME_MAX + WAIT_MAX:
        gap = ring(px, dog.x, w.width)
        v = 0.0
        if stage == "approach":
            if abs(gap) <= stop or t > TAME_MAX:
                stage, t_wait = "wait", 0.0
            elif (t % CYCLE) < duty * CYCLE:
                v = math.copysign(walk, gap)
        else:
            t_wait += 1 / 60
            if abs(gap) <= offer:
                offered = True
                dog.offered(ctx)
            if offered or t_wait > WAIT_MAX:
                break
        px += v / 60
        ctx.person = (px, v)
        ctx.events = []
        phys.follow(dog.x)
        dog.tick(1 / 60, ctx)
        phys.step(1 / 60)
        t += 1 / 60
    success = float(np.clip(0.5 + 6.0 * (dog.trust - trust0), 0.0, 1.0))
    heads.skill.learn(rec, success)
    dog.remove()
    return success
