"""Жизнь человечка без окна: мир, тело, физика, мозг, речь и гитара вместе.

tick() - один шаг физики (1/60 с); каждый второй шаг решает мозг. Возвращает события:
("phrase", текст), ("jump",), ("fall",), ("land",), ("up",), и гитарные: ("pickup",) -
подобрал гитару, ("sit",) - сел играть, ("attempt_start", попытка), ("attempt", запись
журнала), ("stand",) - встал.

Гитара лежит в полутора метрах перед ним, подбирает, проходя рядом, и дальше носит её с
собой. Играть ли - решает общий выбор занятий (acts/choice.py): гитара тянет сильнее мира -
садится, мир снова сильнее (гитара надоела - скука растёт с игрой подряд) - встаёт. Пока играет,
мозг ходьбы не делает шагов и не учится, тело сидит, руки - по попытке.
Окно только рисует и играет звук; самопроверка и кадры README гоняют то же самое.
"""
import math

import numpy as np

from acts.explore.navigator import LOOK_TIME, Walk
from toys.chest import CHEST_AHEAD, REACH, Inventory
from body import skeleton
from body.cppn import generate_body
from body.physics import DT, Human
from brain.brain import ALIVE, FALL_PENALTY, LR, PHYSICS_PER_STEP, PLAY_LR, PLAY_NOISE
from brain.observe import observe
from brain.state import summarize
from music import hands
from acts.choice import DECIDE_EVERY, NEEDS, STICK, Choice, pleasure
from acts.daily import Daily
from acts.swim import Swim
from skills.swim import Swimmer
from acts.eat import FoodPrefs
from acts.needs import Backpack, Needs
from world.food import Food
from world.fish import School
from toys import registry as toy_registry
from world.water import Water
from music.session import FPS
from world.generate import generate

PHRASE_PAUSE = {"rare": 14.0, "normal": 7.0, "often": 3.5}
FELL_MEMORY = 4.0          # с - столько «упал недавно» остаётся в состоянии
RECENT = 8                 # фраз, которые сеть не повторяет подряд
CANDIDATES = 3             # выборок речи на одну фразу
TEMPERATURE = 0.75
LOG_EVERY = 0.5            # с - точка графика любопытства
INTENT_K = 1.0 - math.exp(-DT / 1.0)   # намерение усредняется за секунду
STAND = tuple(skeleton.rest_angles())  # стоит и осматривается
SIT_GROUND = (0.15, 0.1, 0.25, 0.2, 0.7, 0.6, 1.55, 1.5, 0.25, 0.3)   # сидит на земле, ноги вперёд
REST_LYING = 10.0          # с «лежит» за раз - продлевается, пока отдыхает
GUITAR_AHEAD = CHEST_AHEAD  # м - где сундук (и в нём гитара) от места появления
PICKUP = REACH             # м - ближе - дотянулся до сундука или гитары на земле
ATTEMPT_PAUSE = 0.8        # с тишины между попытками
SIT_TIME = 1.0             # с - садится и берёт гитару, прежде чем заиграть
PUT_T = 0.1                # «положить или взять с собой» - температура решения
ASK_T = 0.1                # «положи гитару» - температура согласия
OFFER_BONUS = 0.15         # прибавка к тяге игрушки, которую ему предложили
OFFER_NEAR = 0.6           # м - предложенную кладут рядом, до неё дотянуться
SEEK_MAX = 90.0            # с - столько идёт к положенной гитаре, потом решает заново


class Life:
    def __init__(self, models, world_seed, body_seed, width, brain, frequency="normal", seed=None,
                 guitarist=None, has_guitar=False, guitar_freq="sometimes", choice=None, navigator=None,
                 needs=None, backpack=None, food_prefs=None, swimmer=None, toy_heads=None):
        self.models = models
        self.world_seed, self.body_seed = int(world_seed), int(body_seed)
        self.world = generate(models.world_genome, world_seed, width)
        self.water = Water(self.world)
        self.body = generate_body(models.body_genome, body_seed)
        self.human = Human(self.body, self.world, x=self.dry_spawn(self.world.width * 0.5))
        self.human.water = self.water
        self.fish = School(self.world, self.water)
        self.brain = brain
        self.brain.reset_memory()
        self.brain.opt.lr = LR * PLAY_LR
        self.rng = np.random.default_rng(seed if seed is not None else [world_seed, body_seed])
        self.frequency = frequency
        self.learn = True
        self.nav_learn = None      # None - как learn; отдельно - для обучения навигатора при сборке
        self.noise = PLAY_NOISE    # в игре шум исследования меньше, чем при обучении
        self.last_obs = None       # наблюдение мозга на последнем шаге - для «что видит»
        self.obs_x = self.obs_ground = 0.0
        self.intent = 0.0          # куда хочет идти, среднее за секунду
        self.time = 0.0
        self.phrases = []          # [(время, текст)]
        self.jumps = 0
        self.falls = 0
        self.since_phrase = PHRASE_PAUSE[frequency] * 0.6
        self.fell = 0.0
        self.decision = None
        self.extra = 0.0
        self.k = 0
        self.curiosity_log = []
        self.reward_sum = 0.0
        self.reward_n = 0
        self.speaking = 0.0        # с - рот открыт, пузырь над головой
        self.last_phrase = ""
        self._log_clock = 0.0
        self.guitarist = guitarist
        self.inv = Inventory(self.human.px + CHEST_AHEAD * self.human.facing, self.world.width,
                             toys=[n for n in toy_registry.names() if n != "guitar" or guitarist is not None])
        if has_guitar:
            self.inv.where["guitar"] = "carried"
        if navigator is None:
            from game.models import starter_navigator
            navigator = starter_navigator()
        nav_rng = np.random.default_rng([int(world_seed), int(body_seed), 7] if seed is None else [int(seed), 7])
        self.walk = Walk(self.world.width, self.human.facing, navigator, nav_rng)
        self.choice = choice if choice is not None else Choice(guitar_freq, np.random.default_rng(self.rng.integers(2 ** 31)))
        for t in toy_registry.TOYS.values():
            if t.name != "guitar":
                self.choice.add(t.name, t.half)
        self.toy_heads = toy_heads or {}       # игрушка -> чему она его научила (головы, навыки)
        self.toy = None                        # идущая игра с игрушкой (toys/base.Session)
        self.seek_toy = None                   # за какой игрушкой идёт (к сундуку или где лежит)
        self.seek_since = 0.0
        self.toy_journals = {}
        self.phys = None
        self.decide_clock = 0.0
        self.scores = []           # успех прошлых попыток гитары - для «роста»
        self.playing = False
        self.sitting_for = 0.0
        self.attempt = None
        self.play_t = 0.0
        self.pending_rating = None
        self.last_entry = None
        self.guitar_time = 0.0
        self.music_flags = {}
        self.stopped = False
        self.seeking = None        # с какого времени идёт за гитарой (к сундуку или где положил)
        self.walk.places.marks["сундук"] = self.inv.chest_x
        self.needs = needs if needs is not None else Needs()
        self.backpack = backpack if backpack is not None else Backpack()
        self.food = Food(self.world)
        self.daily = Daily(self, self.needs, self.backpack,
                           food_prefs if food_prefs is not None else FoodPrefs(seed=int(body_seed) % (2 ** 31)), self.food)
        self.posture = "stand"
        self._last_x, self._last_y = self.human.px, self.human.py
        self.swim = Swim(self, swimmer if swimmer is not None else Swimmer(seed=int(body_seed) % (2 ** 31)))
        self.human.can_swim = self.swim.swimmer.can_swim()
        self.swim_flags = {}

    # где гитара: у него, в сундуке или на земле (toys/chest.py)
    @property
    def has_guitar(self):
        return self.inv.where.get("guitar") == "carried"

    @has_guitar.setter
    def has_guitar(self, value):
        if value:
            self.inv.where["guitar"] = "carried"
        elif self.has_guitar:
            self.inv.stow("guitar")

    @property
    def guitar_laid(self):
        return isinstance(self.inv.where.get("guitar"), float)

    @property
    def guitar_x(self):
        """Где лежит гитара на земле, или None."""
        w = self.inv.where.get("guitar")
        return w if isinstance(w, float) else None

    @guitar_x.setter
    def guitar_x(self, x):
        self.inv.where["guitar"] = float(x)

    def dry_spawn(self, x):
        """Появляется на суше: если середина мира под водой - ближайший сухой берег (и сундук
        там же, не в воде)."""
        for d in np.arange(0.0, self.world.width / 2, 0.5):
            for s in (1, -1):
                c = x + s * d
                if all(self.water.depth_at(c + o) <= 0.0 for o in (-CHEST_AHEAD, 0.0, CHEST_AHEAD)):
                    return c % self.world.width
        return x

    @property
    def distance(self):
        return self.human.odometer

    def state(self):
        d = self.decision
        way = 0.0 if self.walk.looking else float(self.walk.want)
        flags = dict(self.need_flags())
        flags.update(self.music_flags)
        return summarize(self.human, self.world, way if d else 0.0, d.want_jump if d else 0.0,
                         self.brain.curiosity, self.fell / FELL_MEMORY, music=flags)

    def say(self):
        s = self.state()
        recent = {t for _, t in self.phrases[-RECENT:]}
        options = [self.models.speech.sample(s, self.rng, TEMPERATURE) for _ in range(CANDIDATES)]
        options = [o for o in options if o.strip()] or ["..."]
        text = next((o for o in options if o not in recent), options[0])
        self.phrases.append((self.time, text))
        self.last_phrase = text
        self.speaking = 2.5
        self.since_phrase = 0.0
        return text

    def stop(self):
        """Жизнь окончена: новый мир, другой человечек, закрытие окна. Идущая попытка
        бросается - без шага обучения и без записи в журнал (законченные уже в журнале и
        сохраняются окном); дальше tick() не делает ничего."""
        self.stopped = True
        self.attempt = None
        self.pending_rating = None
        self.playing = False
        self.music_flags = {}

    def tick(self):
        if self.stopped:
            return []
        self._needs_tick()
        if self.playing:
            return self._guitar_tick()
        events = []
        h = self.human
        if self.swim.mode is not None and self.swim.tick(DT, events):
            return self._water_tick(events)
        if (self.swim.mode is None and h.can_swim and h.grounded
                and self.water.depth_at(h.px + 0.4 * h.facing) > h.wade_depth()):
            self.swim.enter(cross=True)              # пловец: глубокая вода впереди - переплывает
            return self._water_tick(events)
        if self.toy is not None and self._toy_tick(events):
            # замах, удар, бросок: стоит на месте, тело по позе игрушки
            targets = self.toy.pose
            happened = h.step(list(targets), 0.0, False)
            events += [(e,) for e in happened]
            self._clock_tick()
            return events
        if self.daily.task is not None and self.daily.tick(DT, h.px, self.time, events):
            return self._busy_tick(events)
        br = self.brain
        interest = br.last_seen_err / (br.err_ema + 1e-12) if br.err_ema else 1.0
        self.walk.learn = self.learn if self.nav_learn is None else self.nav_learn
        want = self.walk.tick(self.time, DT, h.px, h.facing, interest,
                              can_decide=h.grounded and h.fallen <= 0 and self.decision is not None)
        self.intent += ((0.0 if self.walk.looking else want) - self.intent) * INTENT_K
        left = self.walk.look_left
        # осматривается: сначала оглядывается назад, потом снова вперёд (голова, не тело)
        h.gaze = -h.facing if 0.5 < left < LOOK_TIME - 0.5 else 0
        if self.walk.looking and h.grounded and h.fallen <= 0:
            # стоит и осматривается: мозг ходьбы не шагает, ноги замирают как были (иначе
            # переход в стойку толкает его вперёд), руки опускаются
            targets = h.angles.copy()
            targets[2:6] = STAND[2:6]
            happened = h.step(list(targets), want, False)
            for e in happened:
                events.append((e,))
            self._clock_tick()
            self._world_guitar(events)
            return events
        if self.k % PHYSICS_PER_STEP == 0 or self.decision is None:
            obs = observe(self.human, self.world)
            self.last_obs, self.obs_x = obs, self.human.px
            self.obs_ground = self.world.height_at(self.human.px)
            self.decision, reward = self.brain.step(obs, self.extra, learn=self.learn, noise=self.noise)
            if reward is not None:
                self.reward_sum += reward
                self.reward_n += 1
            d = self.decision
            if self.since_phrase >= PHRASE_PAUSE[self.frequency] and self.rng.random() < d.say:
                events.append(("phrase", self.say()))
            self.extra = 0.0
        d = self.decision
        # куда идти решает навигатор; выход «куда» мозга ходьбы больше не разворачивает
        happened = self.human.step(d.targets, want, d.jump and self.k % PHYSICS_PER_STEP == 0)
        for e in happened:
            events.append((e,))
            if e == "jump":
                self.jumps += 1
            elif e == "fall":
                self.falls += 1
                self.fell = FELL_MEMORY
                self.extra += FALL_PENALTY
        if self.human.grounded and self.human.fallen <= 0 and self.k % PHYSICS_PER_STEP == PHYSICS_PER_STEP - 1:
            self.extra += ALIVE
        self._clock_tick()
        self._world_guitar(events)
        return events

    def _clock_tick(self):
        self.choice.tick(DT)
        self._phys_tick()
        if self.fish.fish:
            splash = self.human.px if self.swim.in_water else None
            self.fish.step(DT, splash)
        self.k += 1
        self.time += DT
        self.since_phrase += DT
        self.fell = max(0.0, self.fell - DT)
        self.speaking = max(0.0, self.speaking - DT)
        self._log_clock += DT
        if self._log_clock >= LOG_EVERY:
            self._log_clock -= LOG_EVERY
            self.curiosity_log.append(float(self.brain.last_err))
            del self.curiosity_log[:-240]

    # --- нужды ---
    def _needs_tick(self):
        h = self.human
        dx = abs(h.px - self._last_x)
        dy = h.py - self._last_y if h.grounded else 0.0
        self._last_x, self._last_y = h.px, h.py
        self.needs.tick(DT, walked=dx, climbed=max(0.0, dy), speed=abs(h.vx), night=self.world.night,
                        posture=self.posture)
        h.stride = self.needs.stride()
        if self.inv.dist(h.px, self.inv.chest_x) < PICKUP and not self.backpack.full():
            self.backpack.refill()              # у сундука рюкзак пополняется

    def _busy_tick(self, events):
        """Занят на месте: ест, пьёт (стоит), отдыхает (сидит или лежит). Мозг ходьбы не шагает."""
        h = self.human
        task = self.daily.task
        posture = task.posture if task is not None else "stand"
        self.posture = posture
        if posture == "lie":
            h.fallen = max(h.fallen, 0.6)        # лежит; вставать - когда отдых кончится
            happened = h.step(None, 0.0, False)
        elif posture == "sit":
            happened = h.step(list(SIT_GROUND), 0.0, False)
        else:
            targets = h.angles.copy()
            targets[2:6] = STAND[2:6]
            if task is not None and task.act in ("eat", "drink"):
                targets[2], targets[4] = 2.0, 2.2   # рука ко рту
            happened = h.step(list(targets), 0.0, False)
        events += [(e,) for e in happened]
        if any(e[0] == "ate" for e in events):
            events.append(("phrase", self.say()))      # вкусно или нет - сразу скажет
            self.daily.flags = {}
        if task is not None and task.act == "rest":
            self.decide_clock += DT
            if self.decide_clock >= DECIDE_EVERY:
                self.decide_clock = 0.0
                if self.choice.choose(self.available(), self.needs.urgency()) != "rest":
                    self.daily.finish(events, pleasure=0.5)
                    self.posture = "stand"
                    if h.fallen > 0:
                        h.fallen = min(h.fallen, 0.5)    # встаёт
                else:
                    task.posture = "lie" if self.needs.tired > 0.75 or task.posture == "lie" else "sit"
        if self.daily.task is None:
            self.posture = "stand"
        self._clock_tick()
        return events

    # --- игрушки (кроме гитары) ---
    def physics(self):
        """pymunk-мир предметов этой жизни - создаётся, когда понадобился."""
        if self.phys is None:
            from physics.world import PhysWorld
            self.phys = PhysWorld(self.world, center=self.human.px)
            self.phys.forces.append(self._water_forces)
        return self.phys

    def _water_forces(self, dt):
        from physics import water as pwater
        for name, (body, shapes) in self.phys.things.items():
            level = self.water.level_at(body.position.x)
            if level is None:
                continue
            s = shapes[0]
            if hasattr(s, "radius") and not hasattr(s, "get_vertices"):
                pwater.apply(body, "circle", s.radius, level, dt)
            else:
                bb = s.bb
                pwater.apply(body, "box", (bb.right - bb.left, bb.top - bb.bottom), level, dt)

    def _phys_tick(self):
        if self.phys is not None and (self.phys.things or self.phys.frozen):
            self.phys.follow(self.human.px)
            self.phys.step(DT)

    def make_session(self, name):
        from toys import base as toy_base
        if name == "ball":
            from toys.ball import BALL_R, BallHeads, BallSession
            heads = self.toy_heads.setdefault("ball", BallHeads(seed=int(self.body_seed) % (2 ** 31)))
            c1, _, prop = toy_base.look(self.body_seed, toy_registry.names().index("ball"))
            if "ball" not in self.toy_journals:
                self.toy_journals["ball"] = toy_base.journal(self.body_seed, "ball")
            journal = self.toy_journals["ball"]
            return BallSession(self, heads, journal, radius=BALL_R[0] + (BALL_R[1] - BALL_R[0]) * prop, color=c1)
        return None

    def toy_visuals(self):
        """Что нарисовать из игрушек: [(вид, x, y, размер, цвет)] - мяч в полёте, у него в руке
        или на земле; метка цели."""
        from toys import base as toy_base
        out = []
        if "ball" in self.inv.where:
            c1, _, prop = toy_base.look(self.body_seed, toy_registry.names().index("ball"))
            from toys.ball import BALL_R
            r = BALL_R[0] + (BALL_R[1] - BALL_R[0]) * prop
            where = self.inv.where["ball"]
            b = self.phys.body("мяч") if self.phys is not None else None
            if b is not None:
                out.append(("ball", float(b.position.x), float(b.position.y), r, c1))
            elif where == "carried":
                pts = self.human.points()
                hx, hy = pts["ha_r" if self.human.facing > 0 else "ha_l"]
                out.append(("ball", hx, hy, r, c1))
            elif isinstance(where, float):
                out.append(("ball", where, self.world.height_at(where) + r, r, c1))
        if self.toy is not None and getattr(self.toy, "target", None) is not None:
            out.append(("target", self.toy.target, self.world.height_at(self.toy.target), 0.0, (0.95, 0.85, 0.3)))
        return out

    def start_toy(self, name, events):
        self.toy = self.make_session(name)
        if self.toy is None:
            self.choice.switch("explore")
            return
        if self.choice.current != name:
            self.choice.switch(name)
        events.append(("toy_start", name))
        self.toy.begin()

    def _toy_tick(self, events):
        """Игра с игрушкой: True - он сейчас на месте (замах, удар), False - ходит (за мячом)."""
        playing = self.toy.tick(DT, events)
        if not playing:
            name = self.toy.name
            self.toy = None
            self.walk.goal = None
            if self.inv.where.get(name) == "carried" and self.wants_to_leave_it(name):
                self.inv.put(name, self.human.px)
                self.walk.places.marks[name] = self.human.px
                events.append(("put_toy", name))
            return False
        return self.toy.pose is not None

    def _water_tick(self, events):
        for e in events:
            if e[0] == "rescued":
                self.swim_flags = {"rescued": 1.0}
            elif e[0] == "ashore":
                if e[1]:
                    self.swim_flags = {"rescued": 1.0}
                    events.append(("phrase", self.say()))
                self.swim_flags = {}
                if self.choice.current == "swim":
                    self.choice.switch("explore")
                self.decide_clock = 0.0
        if self.swim.in_water and not self.swim_flags:
            self.swim_flags = {"swimming": 1.0}
        elif not self.swim.in_water and self.swim_flags.get("swimming"):
            self.swim_flags = {}
        self.posture = "swim"
        self._clock_tick()
        return events

    def after_swim(self, r):
        """Попытка плавания кончилась: удовольствие - в выбор; стал пловцом - глубокая вода открыта."""
        from skills.swim import SWIMMER
        self.choice.after_attempt("swim", pleasure(min(1.0, max(0.0, r) / (2 * SWIMMER)),
                                                   self.swim.swimmer.history[:-1]))
        self.human.can_swim = self.swim.swimmer.can_swim()
        if self.choice.choose(self.available(), self.needs.urgency()) != "swim":
            self.swim.mode = "cross"                 # хватит - к берегу

    def need_flags(self):
        n = self.needs
        f = {"hungry": 1.0 if n.hunger > 0.6 else 0.0, "thirsty": 1.0 if n.thirst > 0.6 else 0.0,
             "tired": 1.0 if n.tired > 0.6 else 0.0}
        f.update(self.daily.flags)
        f.update(self.swim_flags)
        return f

    # --- гитара ---
    def available(self):
        acts = ["explore"]
        if self.guitarist is not None and "guitar" in self.inv.where:
            acts.append("guitar")
        acts += self.daily.available(self.human.px, self.time)
        if self.swim.available(self.human.px):
            acts.append("swim")
        for t in toy_registry.TOYS.values():
            if t.name == "guitar" or t.name not in self.inv.where:
                continue
            if t.needs == "water" and not self.water.ponds:
                continue                         # удочка - только где есть вода
            if self.inv.where[t.name] == "flying":
                continue
            acts.append(t.name)
        return acts

    def _world_guitar(self, events):
        self.choice.world_interest(self.brain.err_ema, DT)
        self.decide_clock += DT
        if self.seek_toy is not None:
            # идёт за игрушкой: взял (из сундука или с земли) - играет
            if self.inv.take(self.seek_toy, self.human.px):
                name, self.seek_toy = self.seek_toy, None
                self.walk.goal = None
                events.append(("pickup_toy", name))
                self.start_toy(name, events)
                return
            if self.time - self.seek_since > SEEK_MAX:
                self.seek_toy = None
                self.walk.goal = None
                self.choice.switch("explore")
            return
        if self.seeking is not None and not self.has_guitar:
            # идёт за гитарой: берёт, когда дотянулся (из сундука или с земли), и садится играть
            if self.inv.take("guitar", self.human.px):
                self.pick_guitar()
                events.append(("pickup",))
                if self.human.grounded and self.human.fallen <= 0:
                    self.sit_down()
                    events.append(("sit",))
                return
            if self.time - self.seeking > SEEK_MAX:
                self.seeking = None                 # не дошёл - решит заново
                self.walk.goal = None
        if (self.seeking is None and self.daily.task is None and self.swim.mode is None and self.toy is None
                and self.decide_clock >= DECIDE_EVERY
                and self.human.grounded and self.human.fallen <= 0 and not self.walk.looking):
            self.decide_clock = 0.0
            pick = self.choice.choose(self.available(), self.needs.urgency())
            if pick in toy_registry.TOYS and pick != "guitar":
                held = self.inv.carried
                if held == pick:
                    self.start_toy(pick, events)
                    return
                if held is not None:
                    self.put_held()
                    events.append(("put_toy", held))
                self.seek_toy, self.seek_since = pick, self.time
                self.walk.goal = self.inv.position(pick)
                if self.walk.goal is None:
                    self.seek_toy = None
                    self.choice.switch("explore")
                return
            if pick == "swim":
                if not self.swim.start(self.human.px):
                    self.choice.switch("explore")
                return
            if pick in NEEDS:
                self.daily.start(pick, self.human.px, self.time)
                if self.daily.task is None:
                    self.choice.switch("explore")
                return
            if pick == "guitar":
                if self.has_guitar:
                    self.sit_down()
                    events.append(("sit",))
                else:
                    # гитара в сундуке или лежит, где он её положил: он идёт к ней (куда - решает
                    # навигатор по памяти мест, метка «гитара» тянет), играть - когда дойдёт
                    self.choice.switch("explore")
                    if self.inv.carried not in (None, "guitar"):
                        events.append(("put_toy", self.inv.carried))
                        self.put_held()
                    self.seeking = self.time
                    self.walk.places.marks["гитара"] = self.inv.position("guitar")
                    self.walk.goal = self.inv.position("guitar")

    def put_held(self):
        """Руки заняты не тем: положить, что держит, здесь же."""
        held = self.inv.carried
        if held == "guitar":
            self.put_guitar()
        elif held is not None:
            self.inv.put(held, self.human.px)
            self.walk.places.marks[held] = self.human.px

    def pick_guitar(self):
        self.inv.where["guitar"] = "carried"
        self.seeking = None
        self.walk.goal = None
        self.walk.pulls.pop("гитара", None)
        self.walk.places.marks.pop("гитара", None)

    def put_guitar(self):
        """Кладёт гитару на землю, где стоит; место - в памяти мест."""
        self.inv.put("guitar", self.human.px)
        self.walk.places.marks["гитара"] = self.human.px
        self.walk.pulls.pop("гитара", None)

    def offer(self, name):
        """«Предложить» из «Каталога»: игрушку кладут рядом с ним. Возьмёт ли - тот же выбор
        занятий с прибавкой OFFER_BONUS; не взял - она лежит рядом (и в памяти мест).
        Возвращает (взял, события) или None, если предложить нечего."""
        if self.stopped or self.playing or name not in self.inv.where or self.inv.where[name] == "carried":
            return None
        held = self.inv.carried
        if held == "guitar":
            self.put_guitar()                    # руки заняты: своё кладёт рядом
        elif held is not None:
            self.inv.put(held, self.human.px)
        x = self.human.px + OFFER_NEAR * self.human.facing
        self.inv.where[name] = float(x)
        self.walk.places.marks["гитара" if name == "guitar" else name] = x
        c = self.choice
        probs = c.probabilities(["explore", name], {name: OFFER_BONUS})
        took = self.rng.random() < probs[name]
        events = []
        if took:
            self.inv.take(name, self.human.px)
            if name == "guitar":
                self.pick_guitar()
                events.append(("pickup",))
                self.sit_down()
                events.append(("sit",))
        self.music_flags = {"offered": 1.0, "agreed": 1.0 if took else 0.0}
        events.append(("phrase", self.say()))
        self.music_flags = {"playing": 1.0} if self.playing else {}
        return took, events

    def wants_to_leave_it(self, name="guitar"):
        """Наигравшись: положить игрушку здесь или взять с собой. Нравится она - берёт с собой
        (скоро захочется снова), тянет гулять далеко - кладёт, чтобы не мешала. Softmax, не правило."""
        c = self.choice
        carry = c.acts[name].liking
        leave = c.acts["explore"].pull()
        return self.rng.random() < 1.0 / (1.0 + math.exp(-(leave - carry) / PUT_T))

    def ask_put(self):
        """«Положи гитару» от владельца - просьба: согласие тем вероятнее, чем меньше ему сейчас
        хочется играть. Возвращает (согласился, события) или None, если гитары у него нет."""
        if not self.has_guitar or self.stopped:
            return None
        c = self.choice
        g = c.acts["guitar"].pull() + (STICK if self.playing else 0.0)
        e = c.acts["explore"].pull()
        agree = self.rng.random() < 1.0 / (1.0 + math.exp((g - e) / ASK_T))
        events = []
        playing = self.playing
        if agree:
            if playing:
                self.attempt = None              # идущая попытка брошена без записи
                self.stand_up()
                events.append(("stand",))
            self.put_guitar()
            events.append(("put",))
        self.music_flags = {"asked_put": 1.0, "agreed": 1.0 if agree else 0.0, "refused": 0.0 if agree else 1.0,
                            "playing": 1.0 if (playing and not agree) else 0.0}
        events.append(("phrase", self.say()))
        self.music_flags = {"playing": 1.0} if self.playing else {}
        return agree, events

    def sit_down(self):
        self.walk.rest(self.human.facing)
        self.playing = True
        if self.choice.current != "guitar":
            self.choice.switch("guitar")
        self.attempt = None
        self.sitting_for = 0.0
        self.music_flags = {"playing": 1.0}

    def stand_up(self):
        self.walk.rest(self.human.facing)
        self.playing = False
        self.attempt = None
        if self.choice.current != "explore":
            self.choice.switch("explore")
        self.decide_clock = 0.0
        self.music_flags = {}
        self.brain.reset_memory()

    def rate(self, rating):
        """👍 (1) или 👎 (-1) от владельца: идущей попытке - в её награду, законченной - после."""
        if self.attempt is not None:
            self.pending_rating = rating
            return None
        if self.guitarist is not None:
            return self.guitarist.rate_last(rating)
        return None

    def _guitar_tick(self):
        events = []
        g = self.guitarist
        h = self.human
        h.step(list(hands.SITTING), 0.0, False)
        self.sitting_for += DT
        if self.attempt is None:
            if self.sitting_for < SIT_TIME:
                self._clock_tick()
                return events
            self.attempt = g.attempt(self.body)
            self.play_t = 0.0
            self.pending_rating = None
            tune = self.attempt.tune
            self.music_flags = {"playing": 1.0, "learning_tune": 1.0 if tune else 0.0}
            events.append(("attempt_start", self.attempt))
        a = self.attempt
        fr = min(len(a.arms) - 1, int(self.play_t * FPS))
        ang = h.angles.copy()
        ang[[2, 4, 3, 5]] = a.arms[fr]            # sh_l, el_l, sh_r, el_r
        h.angles = ang
        self.play_t += DT
        self.guitar_time += DT
        if self.play_t >= a.duration + ATTEMPT_PAUSE:
            entry, progress, novelty = g.finish(a, self.body, rating=self.pending_rating)
            self.last_entry = entry
            # успех для удовольствия: и похоже ли на мелодию, и приятно ли звучит - новичку
            # нравится играть, даже когда на мелодию ещё не похоже
            nice = (entry["B"] + 1.0) / 2.0
            success = nice if entry["A"] is None else 0.5 * (entry["A"] + nice)
            self.choice.after_attempt("guitar", pleasure(success, self.scores, entry.get("rating")))
            self.scores = (self.scores + [success])[-20:]
            best = max([e["A"] for e in g.journal.entries[:-1] if e.get("tune") == entry["tune"]
                        and e.get("phrase") == entry["phrase"] and e.get("A") is not None] or [0.0])
            got = entry["A"] is not None and entry["A"] >= 0.6 and entry["A"] > best
            failing = entry["A"] is not None and entry["A"] < 0.2
            self.music_flags = {"playing": 1.0, "learning_tune": 1.0 if entry["tune"] else 0.0,
                                "got_it": 1.0 if got else 0.0, "failing": 1.0 if failing else 0.0}
            events.append(("attempt", entry))
            if self.since_phrase >= PHRASE_PAUSE[self.frequency] or got:
                events.append(("phrase", self.say()))
            self.attempt = None
            pick = self.choice.choose(self.available(), self.needs.urgency())
            if pick != "guitar":
                self.stand_up()
                self.music_flags = {"stood_up": 1.0}
                events.append(("stand",))
                events.append(("phrase", self.say()))
                self.music_flags = {}
                if self.wants_to_leave_it():
                    self.put_guitar()
                    events.append(("put",))
                if pick in NEEDS:
                    self.choice.switch(pick)
                    self.daily.start(pick, self.human.px, self.time)
        self._clock_tick()
        return events

    def take_reward(self):
        """Средняя награда с прошлого вызова - точка графика обучения в статистике."""
        if not self.reward_n:
            return None
        avg = self.reward_sum / self.reward_n
        self.reward_sum, self.reward_n = 0.0, 0
        return avg
