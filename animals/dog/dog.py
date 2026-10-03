"""Собака целиком: тело (animals/dog/body.py), сеть движения (motor.py), поведение со своей
волей, нужды и доверие к нему (спецификация 2.0, раздел 8).

**Поведение** - как его выбор занятий (acts/choice.py): у каждого занятия собаки (гулять-
нюхать, идти за ним, приносить мяч, плавать, отдыхать) своё «нравится» (медленное среднее
удовольствия) и своя скука (растёт, пока занятие идёт, спадает без него); есть, пить,
отдыхать тянет нужда. Softmax с прибавкой текущему. Так ручная собака сама бросает
приносить мяч или идти за ним - надоело - и потом возвращается; доверие от её скуки не падает.

**Доверие к нему** (0..1): растёт, когда он подходит медленно, стоит рядом и даёт лакомство;
падает, когда он бежит на неё или бросает в неё мячом. Неручная держит расстояние - тем
дальше, чем она пугливее и чем меньше доверяет; бежит на неё - убегает. Доверие выше TAME -
ручная.

**Принести мяч** - своя маленькая сеть (acts/policy.ChoicePolicy): схватив мяч, она решает -
нести ему, убежать с мячом или бросить. Учится по награде «похвалили и дали лакомство, когда
принесла»: сначала убегает с мячом, потом приносит чаще (закон в тестах).

**Не застревает:** продвижения нет - прыгает; нет и дальше - идёт в другую сторону (кольцо -
дойдёт и так); в глубокой воде плывёт. Далеко от окна физики - дешёвый шаг (точка у земли).

Мир вокруг она видит через ctx (DogCtx): мир, вода, физика, где он и куда идёт, мяч.
"""
import math

import numpy as np

from acts.choice import Activity, STICK, T, pleasure
from acts.policy import ChoicePolicy
from animals.dog import motor as motor_mod
from animals.dog.body import DogBody
from animals.dog.look import look as dog_look

ACTS = ("explore", "follow", "fetch", "swim", "chase", "rest", "eat", "drink")
NEEDS = ("rest", "eat", "drink")
HALF = {"explore": 6.0, "follow": 3.0, "fetch": 3.0, "swim": 2.5, "chase": 1.5}   # мин - половина тяги уходит
RECOVER = 3.0               # мин - скука спадает в e раз без занятия
DECIDE = 3.0                # с между точками выбора
MIN_KEEP = 12.0             # с - не меньше в одном занятии, если нужда не срочная
TAME = 0.7                  # доверие выше - ручная
FAST = 1.3                  # м/с - он бежит на неё
SLOW = 0.7                  # м/с - подходит медленно
PATIENT = 0.004             # доверия в секунду, пока он рядом и спокоен
RUN_AT = 0.06               # доверия долой за каждую секунду его бега на неё
TREAT = 0.09                # доверия за лакомство
THROWN_AT = 0.06            # доверия долой, если мяч упал рядом с неручной
GRAB = 0.45                 # м - схватила мяч
DROP_NEAR = 0.9             # м - принесла: бросила у его ног
PRAISE_WAIT = 2.0           # с - не похвалили за это время - «нести» ничего не дало
STUCK_JUMP = 2.5            # с без продвижения - прыжок
STUCK_TURN = 7.0            # с без продвижения - в другую сторону
FETCH_CHOICES = ("bring", "run", "drop")
FETCH_REWARD = {"bring": None, "run": 0.05, "drop": 0.0}      # bring - от него (похвала, лакомство)
NEED_RATE = {"hunger": 1 / 900.0, "thirst": 1 / 700.0, "tired": 1 / 1200.0}
URGE_FROM = 0.45
URGE_K = 2.0


class DogCtx:
    """Что собака знает о мире. Life и тесты дают своё."""

    def __init__(self, world, phys, water=None, time=0.0):
        self.world, self.phys, self.water, self.time = world, phys, water, time
        self.person = None          # (x, vx) или None
        self.ball = None            # (x, y, скорость) мяча, если он лежит или летит в физике
        self.thrown = False         # он только что бросил мяч (сам ставит, собака сбрасывает)
        self.birds = []             # x сидящих птиц
        self.events = []

    def level(self, x):
        return None if self.water is None else self.water.level_at(x)

    def ponds(self):
        return [] if self.water is None else self.water.ponds

    def take_ball(self):
        """Собака схватила мяч: убрать его из физики. Возвращает True, если было что брать."""
        if self.phys.body("мяч") is None:
            return False
        self.phys.remove("мяч")
        return True

    def drop_ball(self, x, y):
        self.phys.add_ball("мяч", x, y, radius=0.11, mass=0.4)


def ring(a, b, width):
    """Знаковое расстояние от a до b по кольцу ширины width."""
    return (b - a + width / 2) % width - width / 2


class DogChoice:
    def __init__(self, look_, rng):
        self.rng = rng
        self.acts = {n: Activity(n, HALF.get(n, 60.0), liking=0.5, need=n in NEEDS) for n in ACTS}
        self.acts["fetch"].liking = 0.3 + 0.5 * look_.playful
        self.acts["follow"].liking = 0.4
        self.acts["swim"].liking = 0.25 + 0.3 * look_.playful
        self.acts["chase"].liking = 0.3 + 0.4 * look_.playful        # гоняться за птицами
        self.current = "explore"
        self.since = 0.0

    def tick(self, dt):
        self.since += dt
        for n, a in self.acts.items():
            if a.need:
                continue
            if n == self.current:
                a.boredom += a.rate() * (1.0 - a.boredom) * dt
            else:
                a.boredom -= a.boredom * dt / (RECOVER * 60.0)

    def after(self, name, p):
        a = self.acts[name]
        a.liking += 0.05 * (p - a.liking)

    def probabilities(self, available, urgency):
        names = list(available)
        z = np.array([self.acts[n].pull(urgency.get(n, 0.0)) + (STICK if n == self.current else 0.0)
                      for n in names]) / T
        z = np.exp(z - z.max())
        return dict(zip(names, z / z.sum()))

    def choose(self, available, urgency):
        urgent = max(urgency.values()) > 0.9 if urgency else False
        if self.current in available and self.since < MIN_KEEP and not urgent:
            return self.current
        p = self.probabilities(available, urgency)
        names = list(p)
        pick = names[int(self.rng.choice(len(names), p=[p[n] for n in names]))]
        if pick != self.current:
            self.current, self.since = pick, 0.0
        return pick

    def to_dict(self):
        return {n: {"liking": a.liking, "boredom": a.boredom} for n, a in self.acts.items()}

    def load(self, d):
        if isinstance(d, dict):
            for n, v in d.items():
                if n in self.acts and isinstance(v, dict):
                    for k in ("liking", "boredom"):
                        x = v.get(k)
                        if isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0:
                            setattr(self.acts[n], k, float(x))
        return self


class Dog:
    def __init__(self, world_seed, phys, x, rng=None, motor=None, fetch=None, facing=1):
        self.world_seed = int(world_seed)
        self.look = dog_look(world_seed)
        self.rng = rng or np.random.default_rng(world_seed)
        self.phys = phys
        self.width = phys.world.width
        self.body = DogBody(phys, x, self.look, facing)
        self.motor = motor or motor_mod.DogMotor(seed=int(world_seed) % (2 ** 31))
        if fetch is None:
            # Adam шагает на lr независимо от величины градиента: на 0.02 за 15 хваток
            # вероятность почти не менялась (замер) - сеть маленькая, шаг крупнее
            fetch = ChoicePolicy(3, len(FETCH_CHOICES), hidden=8, seed=int(world_seed) % (2 ** 31) + 3,
                                 lr=8e-2, entropy=0.05)
            # новичок с мячом чаще убегает: начальная привычка, которую переучивает награда
            fetch.net.items[-1].b[:] = np.array([-0.2, 0.6, 0.0])
            # ждёт около того, что даёт убежать с мячом: учит нести только похвала (замер мутацией:
            # с базой 0.25 «убежать» и «бросить» наказывались, и она училась нести и без похвалы)
            fetch.baseline, fetch.spread = 0.05, 0.15
        self.fetch = fetch
        self.choice = DogChoice(self.look, np.random.default_rng(self.rng.integers(2 ** 31)))
        self.hunger, self.thirst, self.tired = 0.2, 0.2, 0.1
        self.trust = 0.0
        self.tamed = False
        self.free_x = None              # x, пока она за окном физики (дешёвый шаг)
        self.target = None              # x, куда идёт
        self.want = 0.0                 # желаемая скорость, м/с
        self.pose = None
        self.carrying = False
        self.wait_throw = False         # бросила мяч - ждёт, когда его бросят снова
        self.await_praise = 0.0         # с - ждёт похвалы за принесённый мяч
        self.fetch_plan = None          # (решение, запись) - что делает с мячом
        self.fetch_drop_x = None
        self.fleeing = 0.0
        self.task_t = 0.0
        self.decide_t = 0.0
        self.bout_t = motor_mod.BOUT
        self.bout = None                # (запись, x в начале, худший крен, команда)
        self.best = (None, 0.0)         # (расстояние до цели, когда) - против застревания
        self.wander_dir = facing
        self.wander_t = 0.0
        self.leaving = False
        self.brought = 0
        self.ran_off = 0
        self.stuck_turns = 0
        self.last_person = None
        self.flags = {}

    # --- где она ---
    @property
    def x(self):
        return self.free_x if self.free_x is not None else self.body.x

    @property
    def y(self):
        return self.phys.ground(self.x) + 0.5 if self.free_x is not None else self.body.y

    def dist(self, x):
        return ring(self.x, x, self.width)

    # --- нужды ---
    def urgency(self):
        def u(v):
            return 0.0 if v < URGE_FROM else URGE_K * ((v - URGE_FROM) / (1 - URGE_FROM)) ** 2
        return {"eat": u(self.hunger), "drink": u(self.thirst), "rest": u(self.tired)}

    def _needs(self, dt):
        run = min(1.0, abs(self.body.vx()) / motor_mod.RUN) if self.free_x is None else 0.3
        k = 1.4 - 0.8 * self.look.stamina
        self.hunger = min(1.0, self.hunger + NEED_RATE["hunger"] * dt * (1 + run))
        self.thirst = min(1.0, self.thirst + NEED_RATE["thirst"] * dt * (1 + 1.5 * run))
        if self.pose == "lie":
            self.tired = max(0.0, self.tired - dt / 60.0)
        elif self.pose == "sit":
            self.tired = max(0.0, self.tired - dt / 150.0)
        else:
            self.tired = min(1.0, self.tired + NEED_RATE["tired"] * dt * k * (1 + 3 * run))

    # --- доверие ---
    def keep_away(self):
        """Расстояние, ближе которого неручная его не подпускает, м."""
        if self.tamed:
            return 0.0
        return 0.8 + 7.0 * self.look.shy * (1.0 - self.trust)

    def _trust(self, dt, ctx):
        if ctx.person is None:
            return
        px, pvx = ctx.person
        d = self.dist(px)                        # от собаки до него
        toward = -math.copysign(1.0, d) * pvx     # он идёт на неё - плюс
        near = abs(d) < self.keep_away() + 3.0
        if near and toward > FAST:
            self.trust = max(0.0, self.trust - RUN_AT * dt * (0.3 if self.tamed else 1.0))
            if not self.tamed and abs(d) < self.keep_away() + 2.0:
                self.fleeing = 4.0               # убегает
                self.flags = {"dog_fled": 1.0}
        elif near and abs(toward) < SLOW and abs(d) < self.keep_away() + 2.0:
            self.trust = min(1.0, self.trust + PATIENT * dt * (1.0 - 0.5 * self.look.shy))
        if not self.tamed and self.trust >= TAME:
            self.tamed = True
            ctx.events.append(("dog_tamed",))

    def offered(self, ctx):
        """Он протянул лакомство. Возьмёт, если рядом и не боится. Возвращает True, если взяла."""
        if ctx.person is None or self.fleeing > 0:
            return False
        d = abs(self.dist(ctx.person[0]))
        if d > max(1.6, self.keep_away() * 0.6 + 0.9):
            return False
        self.trust = min(1.0, self.trust + TREAT * (1.0 - 0.5 * self.look.shy))
        self.hunger = max(0.0, self.hunger - 0.15)
        ctx.events.append(("dog_treat",))
        if not self.tamed and self.trust >= TAME:
            self.tamed = True
            ctx.events.append(("dog_tamed",))
        return True

    def ball_landed_near(self, x):
        """Мяч упал рядом: неручную это пугает."""
        if not self.tamed and abs(self.dist(x)) < 1.5:
            self.trust = max(0.0, self.trust - THROWN_AT)
            self.fleeing = 3.0

    def praised(self, treat):
        """Он похвалил за принесённый мяч (и дал лакомство, если было)."""
        if self.fetch_plan is None:
            return
        self.await_praise = 0.0
        _, (x, a) = self.fetch_plan
        r = 1.0 if treat else 0.6
        self.fetch.learn(x, None, a, r)
        self.choice.after("fetch", r)
        if treat:
            self.hunger = max(0.0, self.hunger - 0.1)
        self.fetch_plan = None

    # --- выбор ---
    def available(self, ctx):
        acts = ["explore", "rest", "eat", "drink"]
        if ctx.person is not None and (self.tamed or self.trust > 0.3):
            acts.append("follow")
        if self.tamed and (ctx.ball is not None or self.carrying):
            acts.append("fetch")
        if ctx.ponds():
            acts.append("swim")
        if any(abs(self.dist(b)) < 15.0 for b in ctx.birds):
            acts.append("chase")                 # птицы сели рядом - погоняться
        return acts

    def _decide(self, ctx):
        prev = self.choice.current
        pick = self.choice.choose(self.available(ctx), self.urgency())
        if pick != prev:
            self.task_t = 0.0
            if prev in ("follow", "explore", "swim"):
                self.choice.after(prev, 0.6)
        return pick

    # --- шаг ---
    def tick(self, dt, ctx):
        self._needs(dt)
        self._trust(dt, ctx)
        self.choice.tick(dt)
        self.fleeing = max(0.0, self.fleeing - dt)
        if self.await_praise > 0:
            self.await_praise -= dt
            if self.await_praise <= 0 and self.fetch_plan is not None:
                # принесла, а не похвалили - «нести» ничего не дало (иначе без похвалы она всё
                # равно училась нести: «бросить» наказывалось, а «нести» не учитывалось - замер)
                _, (x, a) = self.fetch_plan
                self.fetch.learn(x, None, a, 0.0)
                self.fetch_plan = None
        self.task_t += dt
        self.decide_t += dt
        if ctx.thrown and self.tamed and not self.carrying:
            self.decide_t = max(self.decide_t, DECIDE)   # бросили мяч - решает сразу, бежать ли
        if self.decide_t >= DECIDE or self.choice.current not in self.available(ctx) and not self.carrying:
            self.decide_t = 0.0
            self._decide(ctx)
        self._plan(dt, ctx)
        self._move(dt, ctx)

    def _plan(self, dt, ctx):
        """Куда идти, как быстро и в какой позе - по занятию."""
        act = self.choice.current
        self.pose, self.want = None, motor_mod.WALK
        person = ctx.person[0] if ctx.person is not None else None
        if self.leaving and person is not None:
            self.target = self.x - math.copysign(30.0, self.dist(person))
            self.want = motor_mod.WALK
            return
        if self.fleeing > 0 and person is not None:
            self.target = self.x - math.copysign(12.0, self.dist(person))
            self.want = motor_mod.RUN
            return
        if self.carrying:
            self._carry(ctx, person)
            return
        if person is not None and not self.tamed and abs(self.dist(person)) < self.keep_away():
            # слишком близко - отходит, не убегая
            self.target = self.x - math.copysign(self.keep_away() + 1.0, self.dist(person))
            return
        if act == "follow" and person is not None:
            side = self.x + self.dist(person) - math.copysign(1.2, self.dist(person))
            gap = abs(self.dist(person))
            self.target = side
            self.want = motor_mod.RUN if gap > 6.0 else (motor_mod.WALK if gap > 1.6 else 0.0)
            if self.want == 0.0 and self.tired > 0.4:
                self.pose = "sit"
        elif act == "fetch" and ctx.ball is not None:
            bx, by, bv = ctx.ball
            if ctx.thrown:
                self.wait_throw = False          # бросили - можно бежать
                ctx.thrown = False
            if self.wait_throw:
                # бросила мяч - ждёт нового броска, не хватает тут же снова (замер: 1230 хваток
                # на 60 бросков, и каждая «принесла» - похвала)
                self.target, self.want = bx, 0.0
                self.pose = "sit" if self.tired > 0.3 else None
                return
            self.target, self.want = bx, motor_mod.RUN
            if abs(self.dist(bx)) < GRAB and by - self.phys.ground(bx) < 0.6 and bv < 4.0:
                if ctx.take_ball():
                    self.carrying = True
                    x = np.array([self.trust, self.look.playful, 1.0])
                    a, rec = self.fetch.choose(x, rng=self.rng)
                    self.fetch_plan = (FETCH_CHOICES[a], (x, a))
                    if FETCH_CHOICES[a] == "run":
                        away = -math.copysign(1.0, self.dist(person)) if person is not None else self.body.facing
                        self.fetch_drop_x = self.x + away * float(self.rng.uniform(6.0, 12.0))
                    ctx.events.append(("dog_grab", FETCH_CHOICES[a]))
        elif act == "chase" and ctx.birds:
            b = min(ctx.birds, key=lambda v: abs(self.dist(v)))
            self.target, self.want = self.x + self.dist(b), motor_mod.RUN   # птицу не поймать - закон мира
            if self.task_t > 12.0:
                self.choice.after("chase", 0.6)
                self.choice.since = MIN_KEEP
        elif act == "swim":
            ponds = ctx.ponds()
            if ponds:
                p = min(ponds, key=lambda q: abs(self.dist((q.left + q.right) / 2)))
                mid = self.x + self.dist((p.left + p.right) / 2)
                self.target = mid + 0.8 * (p.right - p.left) / 2 * math.sin(self.task_t / 4.0)
            if self.task_t > 40.0:
                self.choice.after("swim", 0.7)
                self.choice.since = MIN_KEEP
        elif act == "rest":
            self.want = 0.0
            if self.tamed and person is not None and abs(self.dist(person)) > 2.5 and self.task_t < 20:
                self.target, self.want = self.x + self.dist(person), motor_mod.WALK   # ляжет рядом с ним
            else:
                if self.tired > 0.3:
                    self.choice.since = 0.0      # легла - отдыхает, пока не отдохнёт (пугается - встаёт)
                # садится, потом ложится, если устала
                self.pose = "lie" if self.task_t > 3.0 and self.tired > 0.6 else "sit"
        elif act == "eat":
            self.want = 0.0
            self.pose = None
            if self.task_t > 3.0:                # нашла, что съесть (или съела, что он дал)
                self.hunger = max(0.0, self.hunger - 0.5)
                self.choice.after("eat", 0.7)
                self.choice.since = MIN_KEEP
                self.decide_t = DECIDE
        elif act == "drink":
            ponds = ctx.ponds()
            if ponds:
                edges = [q.left - 0.3 for q in ponds] + [q.right + 0.3 for q in ponds]
                e = min(edges, key=lambda v: abs(self.dist(v)))
                self.target = self.x + self.dist(e)
                if abs(self.dist(e)) < 0.6:
                    self.want = 0.0
                    self.thirst = max(0.0, self.thirst - dt / 3.0)
            else:
                self.want = 0.0                  # лужица под кустом
                self.thirst = max(0.0, self.thirst - dt / 4.0)
            if self.thirst < 0.05:
                self.choice.after("drink", 0.7)
                self.choice.since = MIN_KEEP
                self.decide_t = DECIDE
        else:                                   # гуляет, нюхает
            self.wander_t -= dt
            if self.wander_t <= 0:
                self.wander_t = float(self.rng.uniform(6.0, 18.0))
                if self.rng.random() < 0.35:
                    self.wander_dir = -self.wander_dir
            sniff = (self.wander_t % 9.0) < 1.5
            self.target = self.x + self.wander_dir * 5.0
            self.want = 0.0 if sniff else motor_mod.WALK

    def _carry(self, ctx, person):
        plan, rec = self.fetch_plan if self.fetch_plan is not None else ("drop", None)
        self.want = motor_mod.RUN
        if plan == "bring" and person is not None:
            self.target = self.x + self.dist(person)
            if abs(self.dist(person)) < DROP_NEAR:
                self._drop(ctx)
                self.brought += 1
                self.await_praise = PRAISE_WAIT
                ctx.events.append(("dog_brought",))
            return
        if plan == "run" and self.fetch_drop_x is not None:
            self.target = self.fetch_drop_x
            if abs(self.dist(self.fetch_drop_x)) < 0.6:
                self._drop(ctx)
                self.ran_off += 1
                self._learn_fetch("run")
                ctx.events.append(("dog_ran_off",))
            return
        self._drop(ctx)
        self._learn_fetch("drop")

    def _learn_fetch(self, what):
        if self.fetch_plan is not None:
            x, a = self.fetch_plan[1]
            self.fetch.learn(x, None, a, FETCH_REWARD[what])
            self.choice.after("fetch", FETCH_REWARD[what])
            if what != "bring":
                self.fetch_plan = None

    def _drop(self, ctx):
        hx, hy = (self.x + self.body.facing * 0.5 * self.look.length, self.y) if self.free_x is None else (self.x, self.y)
        ctx.drop_ball(hx, max(hy, self.phys.ground(hx) + 0.15))
        self.carrying = False
        self.wait_throw = True
        self.fetch_drop_x = None

    # --- движение ---
    def _move(self, dt, ctx):
        span = getattr(self.phys, "span", None)
        inside = span is None or span[0] + 3.0 < self.x < span[1] - 3.0
        if self.free_x is not None:
            # за окном физики - дешёвый шаг: точка идёт к цели
            if inside:
                x, self.free_x = self.free_x, None
                self.body.place(x)
            else:
                if self.target is not None and self.want > 0:
                    self.free_x += math.copysign(min(self.want, 1.2) * dt, self.dist(self.target))
                return
        elif not inside:
            self.free_x = self.body.x
            self.body.remove()
            return
        b = self.body
        level = ctx.level(b.x)
        b.pose = self.pose if self.want == 0.0 else None
        if self.target is not None and self.want > 0:
            way = 1 if self.dist(self.target) > 0 else -1
            if abs(self.dist(self.target)) < 0.3:
                self.want = 0.0
            else:
                b.turn(way)
        b.drive = 1.0 if self.want > 0 else 0.0
        if self.want == 0.0 and self.pose is None:
            b.pose = "stand"                     # стоит, нюхает, ждёт
        self._motor(dt, level)
        b.step(dt, level)
        self._unstick(dt)

    def _motor(self, dt, level):
        b = self.body
        self.bout_t += dt
        if self.bout is not None:
            rec, x0, worst, cmd, want = self.bout
            if want > 0:
                g = math.atan2(self.phys.ground(b.x + b.L / 2) - self.phys.ground(b.x - b.L / 2), b.L)
                worst = max(worst, abs((b.angle() - g + math.pi) % (2 * math.pi) - math.pi))
            self.bout = (rec, x0, worst, cmd, want)
        if self.bout_t < motor_mod.BOUT:
            return
        self.bout_t = 0.0
        if self.bout is not None:
            rec, x0, worst, cmd, want = self.bout
            if want > 0 and b.wet < 0.3:
                r = motor_mod.DogMotor.reward(b.facing * (b.x - x0) * (motor_mod.BOUT - motor_mod.SETTLE) / motor_mod.BOUT,
                                              want, worst, cmd)
                self.motor.learn(rec, r)
        slope = motor_mod.slope_ahead(self.phys.world, b.x, b.facing)
        feats = self.motor.features(slope, max(self.want, motor_mod.WALK), b.wet, self.tired)
        cmd, rec = self.motor.act(feats)
        b.gait = cmd
        self.bout = (rec, b.x, 0.0, cmd, self.want)

    def _unstick(self, dt):
        """Не застревает: нет продвижения туда, куда идёт, - прыгает, потом идёт другой дорогой.
        Продвижение - своё смещение в сторону цели (цель «гулять» едет вместе с ней)."""
        if self.want <= 0 or self.target is None or self.body.wet > 0.3:
            self.best = (None, 0.0)
            return
        way = 1 if self.dist(self.target) > 0 else -1
        anchor, since = self.best
        if anchor is None or anchor[1] != way or way * self.dist(anchor[0]) < -0.5:
            self.best = ((self.x, way), 0.0)
            return
        since += dt
        self.best = (anchor, since)
        if since > STUCK_TURN:
            self.best = (None, 0.0)
            self.stuck_turns += 1
            if self.choice.current == "explore":
                self.wander_dir = -self.wander_dir
            else:
                self.target = self.x - way * 4.0
        elif since > STUCK_JUMP and int(since * 2) != int((since - dt) * 2):
            self.body.jump(0.7)

    # --- рисование и сохранение ---
    def visuals(self):
        """(вид собаки, ноги, мяч в зубах?, отряхивается?) для окна."""
        if self.free_x is not None:
            return None
        corners, legs = self.body.points()
        return {"corners": corners, "legs": legs, "head": self.body.head(), "facing": self.body.facing,
                "look": self.look, "carrying": self.carrying, "shake": self.body.shake, "pose": self.pose,
                "wet": self.body.wet, "angle": self.body.angle()}

    def to_dict(self):
        return {"trust": self.trust, "tamed": self.tamed, "hunger": self.hunger, "thirst": self.thirst,
                "tired": self.tired, "choice": self.choice.to_dict(), "brought": self.brought, "ran_off": self.ran_off}

    def load(self, d):
        if not isinstance(d, dict):
            return self
        for k in ("trust", "hunger", "thirst", "tired"):
            v = d.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                setattr(self, k, float(v))
        if isinstance(d.get("tamed"), bool):
            self.tamed = d["tamed"]
        for k in ("brought", "ran_off"):
            v = d.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                setattr(self, k, v)
        self.choice.load(d.get("choice"))
        return self

    def remove(self):
        self.body.destroy()
