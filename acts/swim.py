"""Занятие «плавать» и переправа через глубокую воду.

Плавать он выбирает сам (тот же выбор занятий), если видел водоём глубже брода: идёт к его
берегу, входит в воду и плывёт попытками по ATTEMPT с (навык skills/swim.py решает, как
грести в эту попытку; тело в воде - body/swim.py). Доплыл до другого берега - поворачивает.

Безопасность (закон): он никогда не тонет. Если устал (выше TIRED_OUT), или голова под
водой дольше UNDER с, или пловец из него ещё никакой, а вода глубокая - он ложится на воду, и
его несёт к ближайшему берегу: не дольше RESCUE с. В чат - что выбрался.

Переправа: если он уже пловец (навык), глубокая вода для ходьбы ему открыта - дошёл до неё,
переплывает и идёт дальше.
"""
import math

from body.swim import Stroke, SwimBody
from skills.swim import ATTEMPT

TIRED_OUT = 0.85
UNDER = 2.5              # с головой под водой - хватит (при 1 с новичок не успевал ничему научиться)
RESCUE = 5.0             # с - на берегу не позже
TIRE = 0.004             # усталости в секунду плавания
THIRST = 0.002
FLOAT = Stroke(0.0, 1.0, 0.0, 0.0)


class Swim:
    def __init__(self, life, swimmer):
        self.life = life
        self.swimmer = swimmer
        self.body = None             # SwimBody, пока он в воде
        self.pond = None
        self.mode = None             # "go", "swim", "cross", "rescue"
        self.goal = None
        self.stroke, self.record = FLOAT, None
        self.t = 0.0
        self.under = 0.0
        self.above = 0
        self.steps = 0
        self.start_x = 0.0
        self.rescue_from = None
        self.attempts = 0

    @property
    def in_water(self):
        return self.body is not None

    # --- что можно ---
    def deep_ponds(self):
        w = self.life.water
        wade = self.life.human.wade_depth()
        return [p for p in w.ponds if p.depth > wade]

    def known_deep(self, x):
        places = self.life.walk.places
        best = None
        for p in self.deep_ponds():
            for edge, inward in ((p.left, 1), (p.right, -1)):
                spot = (edge - inward * 0.3) % places.width
                if places.visits[places.index(spot)] == 0:
                    continue
                d = min(places.distance(x, spot, 1), places.distance(x, spot, -1))
                if best is None or d < best[1]:
                    best = (spot, d, p, inward)
        return best

    def available(self, x):
        return self.known_deep(x) is not None

    # --- начать ---
    def start(self, x):
        k = self.known_deep(x)
        if k is None:
            return False
        self.goal, _, self.pond, _ = k
        self.mode = "go"
        self.life.walk.goal = self.goal
        return True

    def enter(self, cross=False):
        h = self.life.human
        p = self.life.water.pond_at(h.px) or self.pond
        if p is None:
            return False
        self.pond = p
        # заходит вброд, пока не станет глубоко, - туда, где глубже
        wade = h.wade_depth()
        x, facing = h.px, h.facing
        if not cross and self.life.water.depth_at(x + 0.5 * facing) < self.life.water.depth_at(x - 0.5 * facing):
            facing = -facing                     # плавать - туда, где глубже; переправа - куда шёл
        for _ in range(200):
            if self.life.water.depth_at(x) > wade * 0.9:
                break
            x += 0.1 * facing
        self.body = SwimBody(x, p.level, facing)
        self.mode = "cross" if cross else "swim"
        self.life.walk.goal = None
        self.new_attempt()
        return True

    def new_attempt(self):
        h = self.life.human
        depth = self.life.water.depth_at(h.px)
        if self.mode == "cross":
            self.swimmer.explore = False          # переправа - как умеет, без проб
        self.stroke, self.record = self.swimmer.plan(depth, self.life.needs.tired)
        self.swimmer.explore = True
        self.t = 0.0
        self.under = 0.0
        self.above = self.steps = 0
        self.start_x = self.body.x
        self.start_path = self.body.path

    def stop(self):
        self.mode = None
        self.body = None
        self.life.walk.goal = None

    # --- шаг ---
    def tick(self, dt, events):
        """Шаг в воде. True - он в воде (ходьбы нет)."""
        life = self.life
        h = life.human
        if self.mode == "go":
            if life.water.depth_at(h.px) > 0.05 or life.inv.dist(h.px, self.goal) < 0.4:
                if life.water.pond_at(h.px + 0.5 * h.facing) is None and life.water.pond_at(h.px) is None:
                    h.facing = -h.facing if life.water.pond_at(h.px - 0.5 * h.facing) else h.facing
                self.enter()
            return False
        if self.body is None:
            return False
        b = self.body
        if self.mode == "rescue":
            self._rescue_step(dt, events)
            if self.body is None:
                return False                     # уже на берегу
        else:
            head = b.step(self.stroke, dt)
            self.steps += 1
            self.above += head > 0
            self.under = self.under + dt if head <= 0 else 0.0
            self.t += dt
            life.needs.tired = min(1.0, life.needs.tired + TIRE * dt)
            life.needs.thirst = min(1.0, life.needs.thirst + THIRST * dt)
            if self.under > UNDER or life.needs.tired > TIRED_OUT:
                self.start_rescue(events)
            elif self.t >= ATTEMPT and self.mode == "swim":
                self.end_attempt(events)
        self.place_body()
        # вышел на мелкое - на берегу
        shallow = life.water.depth_at(b.x) <= h.wade_depth() * 0.8
        toward_shore = life.water.depth_at(b.x + 0.3 * b.facing) < life.water.depth_at(b.x - 0.3 * b.facing)
        if self.mode != "rescue" and shallow and toward_shore:
            if self.mode == "swim":
                b.facing = -b.facing              # доплыл до берега - назад, в воду
                b.x += b.facing * 0.3
                b.vx = 0.0
            else:
                self.leave(events)
                return False
        return True

    def end_attempt(self, events, rescue=False):
        b = self.body
        dist = b.path - self.start_path
        r = self.swimmer.finish(self.record, dist, self.above / max(self.steps, 1), b.effort, max(self.t, 0.5))
        b.effort = 0.0
        self.attempts += 1
        events.append(("swim", dist, r))
        self.life.after_swim(r)
        if self.mode == "swim" and not rescue:
            self.new_attempt()

    def start_rescue(self, events):
        if self.mode == "swim":
            self.end_attempt(events, rescue=True)   # неудачная попытка - тоже урок
        self.mode = "rescue"
        self.rescue_from = self.body.x
        self.stroke = FLOAT
        events.append(("rescued",))

    def _rescue_step(self, dt, events):
        b = self.body
        p = self.pond
        w = self.life.world.width
        span = (p.right - p.left) % w
        off = (b.x - p.left) % w
        target = b.x - off - 0.3 if off < span / 2 else b.x + (span - off) + 0.3
        need = abs(target - self.rescue_from)
        speed = max(0.5, need / (RESCUE - 1.0))
        step = math.copysign(min(speed * dt, abs(target - b.x)), target - b.x)
        b.x += step
        b.head = max(b.head, 0.05)
        if abs(target - b.x) < 1e-6 or self.life.water.depth_at(b.x) <= 0.0:
            self.leave(events)

    def place_body(self):
        """Человечек там, где тело в воде: лежит на воде, руки гребут по фазе."""
        h = self.life.human
        b = self.body
        if b is None:
            return
        h.px = b.x % self.life.world.width
        h.facing = b.facing
        lean = self.stroke.lean if self.mode != "rescue" else 0.0
        h.tilt = -(math.pi / 2 - lean) * 0.95
        h.py = b.level - 0.05 + b.head
        ph = 2 * math.pi * b.phase
        a = h.angles.copy()
        a[2] = 1.5 + 1.6 * math.sin(ph) * self.stroke.arm          # плечи - гребок
        a[3] = 1.5 - 1.6 * math.sin(ph) * self.stroke.arm
        a[4], a[5] = 0.3, 0.3
        a[6] = 0.2 * math.sin(2 * ph) * self.stroke.kick
        a[7] = -a[6]
        a[8] = a[9] = 0.15
        h.angles = a
        h.vx = b.vx * b.facing
        h.vy = 0.0
        h.grounded = False

    def leave(self, events):
        """На берегу: встаёт и идёт."""
        h = self.life.human
        b = self.body
        rescued = self.mode == "rescue"
        self.body = None
        self.mode = None
        h.tilt = 0.0
        h.px = b.x % self.life.world.width
        h.facing = b.facing
        h.py = self.life.world.height_at(h.px) + h.body.leg() + 0.05
        h.vx = h.vy = 0.0
        from body import skeleton
        rest = skeleton.rest_angles()
        h.angles, h.command, h.targets = rest.copy(), rest.copy(), rest.copy()
        h.grounded = False
        events.append(("ashore", rescued))
