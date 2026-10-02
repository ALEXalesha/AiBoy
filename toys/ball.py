"""Мяч: пнуть или бросить в цель, бросить вверх и поймать.

Попытка - одно из трёх (что - выбирает маленькая сеть видов по удовольствию, acts/policy):
- **пнуть в цель** и **бросить в цель**: цель - метка на земле в 3-14 м; голова вида по
  расстоянию до цели решает угол и силу и учится на своём опыте (skills/base.ExperienceSkill:
  «мяч упал там-то - какие были угол и сила»; REINFORCE по промаху за 300 попыток так и не
  стал ровным - замер в плане); успех - насколько близко мяч впервые коснулся земли к метке;
- **бросить вверх и поймать**: голова решает угол и силу броска (REINFORCE: поймал - и чем
  выше бросил, тем лучше); куда мяч опустится,
  угадывает навык «куда прилетит» (skills/predict.py, учится на каждом увиденном полёте), он
  идёт туда и ловит; успех - поймал (или насколько близко был).
Сила тела - его: мяч летит с той скоростью, что дала нога или рука этого человечка (длиннее
нога - сильнее удар); голова учится под своё тело. Мяч - тело pymunk (physics/world.py):
скачет, катится, тонет в пруду не тонет - плавает (physics/water.py). Потом он идёт за мячом.
"""
import math

import numpy as np

from acts.policy import ChoicePolicy
from skills.base import ExperienceSkill, GaussianSkill
from skills.predict import Predictor
from toys.base import Session

KINDS = ("kick", "throw", "catch")
KIND_TITLES = {"kick": "пнуть в цель", "throw": "бросить в цель", "catch": "бросить и поймать"}
TARGET = (3.0, 14.0)
WIND_UP = 0.35           # с замах
REST_V = 0.3             # м/с - мяч остановился
FLIGHT_MAX = 8.0         # с - дольше не следит
REACH = 0.7              # м - дотянулся до мяча
CATCH_R = 0.5            # м - поймал, если мяч прошёл ближе к руке
BALL_R = (0.09, 0.14)


class BallHeads:
    """Всё, чему учится мяч у этого человечка: виды, по голове на вид, предсказатель."""

    def __init__(self, seed=0):
        self.kind = ChoicePolicy(1, len(KINDS), hidden=8, seed=seed, lr=5e-3, entropy=0.2)
        self.kick = ExperienceSkill(1, [0.05, 2.0], [1.2, 20.0], hidden=24, seed=seed + 1, sigma=0.04)
        self.throw = ExperienceSkill(1, [0.05, 2.0], [1.3, 16.0], hidden=24, seed=seed + 2, sigma=0.04)
        self.catch = GaussianSkill(1, [0.9, 3.0], [1.55, 10.0], hidden=12, seed=seed + 3, sigma=0.2)
        self.predict = Predictor(seed=seed + 4)

    def parts(self):
        return {"kick": self.kick, "throw": self.throw, "catch": self.catch}

    def save(self, path):
        from nn import io
        arrays = {}
        for name, part in self.parts().items():
            for k, v in io.collect(part.nets()).items():
                arrays[f"{name}.{k}"] = v
            if hasattr(part, "baseline"):
                arrays[f"{name}.meta"] = np.array([part.baseline, part.spread, part.attempts], np.float64)
        for k, v in io.collect({"net": self.kind.net}).items():
            arrays["kind." + k] = v
        for k, v in io.collect({"net": self.predict.net}).items():
            arrays["predict." + k] = v
        arrays["predict.flights"] = np.array([self.predict.flights])
        io.save(path, arrays)

    def load(self, path):
        from nn import io
        arrays = io.load(path)
        for name, part in self.parts().items():
            io.restore(part.nets(), {k[len(name) + 1:]: v for k, v in arrays.items() if k.startswith(name + ".")})
            if f"{name}.meta" in arrays:
                b, s, n = arrays[f"{name}.meta"]
                part.baseline, part.spread, part.attempts = float(b), float(s), int(n)
        io.restore({"net": self.kind.net}, {k[5:]: v for k, v in arrays.items() if k.startswith("kind.")})
        io.restore({"net": self.predict.net}, {k[8:]: v for k, v in arrays.items() if k.startswith("predict.")
                                               and k != "predict.flights"})
        self.predict.flights = int(arrays["predict.flights"][0])
        return self


class BallSession(Session):
    name = "ball"

    def __init__(self, life, heads, journal_, radius=0.11, color=(0.9, 0.3, 0.2)):
        super().__init__(life, heads, journal_)
        self.radius = radius
        self.color = color
        self.stage = None
        self.t = 0.0
        self.kind = None
        self.kind_record = None
        self.record = None
        self.target = None
        self.start_x = 0.0
        self.landed = None
        self.flight_start = None
        self.best_miss = None
        self.catch_goal = None
        self.trace = []
        self.pose = None             # цели суставов, пока замах и удар; None - ходит
        self.last_vy = 0.0
        self.contact = False

    # --- попытка ---
    def start_attempt(self):
        life = self.life
        h = life.human
        self.kind, _ = self.head.kind.choose(np.ones(1), rng=life.rng)
        self.kind_name = KINDS[self.kind]
        self.stage = "windup"
        self.t = 0.0
        self.start_x = h.px
        self.landed = None
        self.best_miss = None
        self.trace = []
        if self.kind_name == "catch":
            self.target = None
            cmd, self.record = self.head.catch.act(np.ones(1, np.float32))
        else:
            d = float(life.rng.uniform(*TARGET))
            self.target = h.px + h.facing * d
            part = self.head.kick if self.kind_name == "kick" else self.head.throw
            cmd = part.act(np.array([d / 15.0], np.float32))
            self.record = None
        self.apex = 0.0
        self.cmd = (float(cmd[0]), float(cmd[1]))
        self.pose = self._pose(0.0)

    def _pose(self, k):
        """Замах (k 0..1) и сам удар/бросок: цели суставов."""
        a = self.life.human.angles.copy()
        if self.kind_name == "kick":
            a[6] = -0.6 + 1.9 * k            # бедро назад - вперёд
            a[8] = 1.2 * (1 - k)
        else:
            a[3] = 2.6 - 1.4 * k if self.kind_name == "throw" else 2.2 + 0.6 * k
            a[5] = 1.0 * (1 - k)
        return a

    def strength(self):
        b = self.life.body
        if self.kind_name == "kick":
            return b.leg() / 0.85
        return (b.upper_arm + b.forearm) / 0.55 * 0.8

    def release(self, events):
        life = self.life
        h = life.human
        angle, speed = self.cmd
        v = speed * self.strength()
        hand_y = h.py + (0.0 if self.kind_name == "kick" else 0.5) - (h.body.leg() - 0.12 if self.kind_name == "kick" else 0)
        x0 = h.px + h.facing * 0.35
        y0 = max(life.world.height_at(x0) + self.radius + 0.06, hand_y)
        phys = life.physics()
        phys.remove("мяч")
        phys.add_ball("мяч", x0, y0, radius=self.radius, mass=0.4)
        phys.body("мяч").velocity = (h.facing * v * math.cos(angle), v * math.sin(angle))
        life.inv.where["ball"] = "flying"
        self.flight_start = (x0, y0, h.facing * v * math.cos(angle), v * math.sin(angle))
        events.append(("sound", "kick" if self.kind_name == "kick" else "throw", min(1.0, v / 15.0), self.radius))
        self.stage = "flight"
        self.t = 0.0
        self.pose = None
        self.last_vy = v * math.sin(angle)
        if self.kind_name == "catch":
            hand = h.py + 0.6
            dx, _ = self.head.predict.predict(self.flight_start[2] * h.facing, self.flight_start[3], y0 - hand)
            self.catch_goal = h.px + h.facing * dx
            life.walk.goal = self.catch_goal

    def attempt_tick(self, dt, events):
        life = self.life
        h = life.human
        if self.stage == "windup":
            self.t += dt
            self.pose = self._pose(min(1.0, self.t / WIND_UP))
            if self.t >= WIND_UP:
                self.release(events)
            return None
        phys = life.physics()
        b = phys.body("мяч")
        if self.stage == "flight":
            self.t += dt
            x, y = b.position
            vx, vy = b.velocity
            self.apex = max(self.apex, y - self.flight_start[1])
            self.trace.append((round(float(x - self.start_x), 3), round(float(y), 3)))
            ground = life.world.height_at(x) + self.radius + 0.07
            touching = y <= ground
            if touching and not self.contact and self.last_vy < -0.8:
                events.append(("sound", "bounce", min(1.0, -self.last_vy / 8.0), self.radius))
                if self.landed is None:
                    self.landed = x
            self.contact = touching
            self.last_vy = vy
            if self.kind_name == "catch":
                hand = h.py + 0.6
                if vy < 0 and y <= hand + 0.2 and self.landed is None:
                    miss = abs(x - h.px)
                    self.best_miss = miss if self.best_miss is None else min(self.best_miss, miss)
                    if miss < CATCH_R:
                        events.append(("sound", "catch", 0.8, self.radius))
                        self._saw(x, hand)
                        phys.remove("мяч")
                        life.inv.where["ball"] = "carried"
                        life.walk.goal = None
                        return self._result(True)
                if self.landed is not None and self.best_miss is not None and self._saw_once():
                    self._saw(self.landed, life.world.height_at(self.landed))
            speed = math.hypot(vx, vy)
            if (self.landed is not None and speed < REST_V) or self.t > FLIGHT_MAX:
                if self.kind_name == "catch" and self.landed is not None and not self._saw_done:
                    self._saw(self.landed, life.world.height_at(self.landed))
                self.stage = "fetch"
                life.inv.where["ball"] = float(x)
                life.walk.goal = float(x)
                return self._result(False)
        return None

    _saw_done = False

    def _saw_once(self):
        return not self._saw_done

    def _saw(self, x, at_y):
        if self._saw_done or self.flight_start is None:
            return
        x0, y0, vx, vy = self.flight_start
        f = self.life.human.facing
        self.head.predict.saw(vx * f, vy, y0 - at_y, 0.0, (x - x0) * f, self.t)
        self._saw_done = True

    def _result(self, caught):
        land = self.landed
        part = self.head.parts()[self.kind_name]
        if self.kind_name == "catch":
            near = 1.0 if caught else math.exp(-(self.best_miss if self.best_miss is not None else 3.0) / 1.0)
            success = near * (0.5 + 0.5 * min(1.0, self.apex / 5.0))
            part.learn(self.record, success)
        else:
            miss = abs(land - self.target) if land is not None else 15.0
            success = math.exp(-miss / 2.0)
            if land is not None:
                part.remember(np.array([abs(land - self.start_x) / 15.0], np.float32), np.array(self.cmd, np.float32))
                part.learn()
        self.head.kind.learn(np.ones(1), None, self.kind, success)
        self._saw_done = False
        entry = {"kind": self.kind_name, "target": None if self.target is None else round(self.target - self.start_x, 2),
                 "landed": None if land is None else round(land - self.start_x, 2), "caught": bool(caught),
                 "angle": round(self.cmd[0], 3), "speed": round(self.cmd[1], 2), "trace": self.trace[::3]}
        return success, entry

    def ready(self):
        return self.life.inv.where.get("ball") == "carried"

    def tick(self, dt, events):
        """Между попытками он идёт за мячом и подбирает его."""
        if not self.active:
            return False
        if self.stage == "fetch":
            life = self.life
            x = life.inv.where.get("ball")
            b = life.physics().body("мяч")
            if b is not None:
                x = float(b.position.x)
                life.inv.where["ball"] = x
                life.walk.goal = x
            if isinstance(x, float) and life.inv.dist(life.human.px, x) < REACH:
                life.physics().remove("мяч")
                life.inv.where["ball"] = "carried"
                life.walk.goal = None
                self.start_attempt()
            return True
        return super().tick(dt, events)


class _Flat:
    width, dx = 2000.0, 0.25

    def height_at(self, x):
        return 0.0


def practice(heads, kind, rng, strength=1.0, world=None):
    """Одна попытка без жизни вокруг (для замеров и тестов): мяч из руки или от ноги на ровной
    земле, полёт в pymunk до первого касания. Возвращает успех 0..1."""
    from physics.world import PhysWorld
    world = world or _Flat()
    p = PhysWorld(world, center=0.0)
    if kind == "catch":
        cmd, rec = heads.catch.act(np.ones(1, np.float32))
        target = None
    else:
        d = float(rng.uniform(*TARGET))
        target = d
        part = heads.kick if kind == "kick" else heads.throw
        cmd = part.act(np.array([d / 15.0], np.float32))
    angle, speed = float(cmd[0]), float(cmd[1])
    v = speed * strength
    y0 = 0.2 if kind == "kick" else 1.5
    p.add_ball("мяч", 0.0, y0)
    b = p.body("мяч")
    b.velocity = (v * math.cos(angle), v * math.sin(angle))
    landed, t, apex = None, 0.0, 0.0
    hand = 1.4
    while t < FLIGHT_MAX:
        p.follow(b.position.x)
        p.step(1 / 60)
        t += 1 / 60
        apex = max(apex, b.position.y - y0)
        if kind == "catch" and b.velocity.y < 0 and b.position.y <= hand:
            landed = b.position.x
            break
        if b.position.y <= 0.11 + 0.07 and b.velocity.y <= 0:
            landed = b.position.x
            break
    if kind == "catch":
        guess, _ = heads.predict.predict(v * math.cos(angle), v * math.sin(angle), y0 - hand)
        miss = abs(guess - (landed or 0.0))
        heads.predict.saw(v * math.cos(angle), v * math.sin(angle), y0 - hand, 0.0, landed or 0.0, t)
        success = (1.0 if miss < CATCH_R else math.exp(-miss)) * (0.5 + 0.5 * min(1.0, apex / 5.0))
        heads.catch.learn(rec, success)
    else:
        miss = abs((landed or 0.0) - target)
        success = math.exp(-miss / 2.0)
        part.remember(np.array([(landed or 0.0) / 15.0], np.float32), np.array([angle, speed / strength], np.float32))
        part.learn()
    return success
