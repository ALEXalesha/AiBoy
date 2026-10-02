"""Есть, пить, отдыхать - занятия нужд. Их, как и всё, выбирает общий выбор (acts/choice.py):
нужда - это срочность, прибавка к тяге занятия; здесь - что происходит, когда выбрал.

- **есть**: к какой еде идти, решает его вкус (acts/eat.py) - известная ему еда мира (места,
  где он был, не дальше SEE м) или рюкзак; дошёл - ест EAT_TIME с, голод уходит, вкус -
  удовольствие; вкус учится на этом;
- **пить**: к воде, если видел пруд не дальше SEE м (там же наполняет флягу), иначе из
  фляги; пустая и воды не видел - к сундуку;
- **отдыхать**: сидит, а если устал сильно - ложится; лёжа проходит быстрее всего; встаёт,
  когда другое занятие тянет сильнее (выбор раз в DECIDE_EVERY);
- у сундука рюкзак пополняется.
"""
import numpy as np

from acts import eat as eat_mod
from world.food import PORTION

EAT_TIME = 3.0
DRINK_TIME = 2.0
SIP = 0.35                  # фляги за раз
DRINK = 0.6                 # жажды уходит за раз
BACKPACK_PLEASURE = 0.1     # еда из рюкзака: сытно, но без радости
LIE_FROM = 0.75             # усталость выше - ложится, ниже - садится
GO_MAX = 90.0               # с - столько идёт к цели, потом решает заново
REACH = 0.8
SEE = eat_mod.SEE


class Task:
    def __init__(self, act, target=None, item=None, stage="go"):
        self.act, self.target, self.item, self.stage = act, target, item, stage
        self.left = 0.0
        self.started = 0.0
        self.posture = "stand"


class Daily:
    def __init__(self, life, needs, backpack, prefs, food):
        self.life = life
        self.needs, self.backpack, self.prefs, self.food = needs, backpack, prefs, food
        self.task = None
        self.flags = {}
        self.last_meal = None          # (вкус, что) - для чата и тестов

    # --- что можно выбрать ---
    def known_food(self, x, t):
        places = self.life.walk.places
        out = []
        for i, e in self.food.available(t):
            if places.visits[places.index(e.x)] == 0:
                continue                         # не видел - не знает
            d = places.distance(x, e.x, 1)
            d = min(d, places.width - d)
            if d <= SEE:
                out.append((i, self.food.feats[i], d))
        return out

    def available(self, x, t):
        acts = ["rest", "drink"]                  # пустая фляга - к сундуку за водой
        if self.known_food(x, t) or self.backpack.food > 0:
            acts.append("eat")
        return acts

    def known_shore(self, x):
        """(точка у воды, сколько идти) - ближайший берег пруда, который он видел, или None."""
        water = getattr(self.life, "water", None)
        if water is None:
            return None
        places = self.life.walk.places
        best = None
        for p in water.ponds:
            for edge, inward in ((p.left, 1), (p.right, -1)):
                spot = edge - inward * 0.3                # на берегу, у самой воды
                if places.visits[places.index(spot)] == 0:
                    continue
                d = places.distance(x, spot, 1)
                d = min(d, places.width - d)
                if d <= SEE and (best is None or d < best[1]):
                    best = (spot % places.width, d)
        return best

    def food_choices(self, x, t):
        """{что: вероятность пойти есть это}: известная еда мира и рюкзак (без дороги, без
        радости) - softmax по тому, сколько удовольствия он ждёт, минус дорога."""
        keys, z = [], []
        for i, f, d in self.known_food(x, t):
            keys.append(i)
            z.append(self.prefs.expect(f) - eat_mod.WAY_COST * d)
        if self.backpack.food > 0:
            keys.append("рюкзак")
            z.append(BACKPACK_PLEASURE)
        if not keys:
            return {}
        z = np.exp((np.array(z) - max(z)) / eat_mod.T)
        return dict(zip(keys, z / z.sum()))

    def pick_food(self, x, t, rng):
        probs = self.food_choices(x, t)
        if not probs:
            return None
        keys = list(probs)
        return keys[int(rng.choice(len(keys), p=[probs[k] for k in keys]))]

    # --- начать ---
    def start(self, act, x, t):
        life = self.life
        if act == "eat":
            key = self.pick_food(x, t, life.rng)
            if key is None:
                return None
            if key == "рюкзак":
                task = Task("eat", None, "рюкзак", stage="do")
            else:
                task = Task("eat", self.life.world.entities[key].x, key)
        elif act == "drink":
            shore = self.known_shore(x)
            if shore is not None and (self.backpack.flask <= 1e-6 or shore[1] < SEE / 2):
                task = Task("drink", shore[0], "вода")
            elif self.backpack.flask > 1e-6:
                task = Task("drink", None, "фляга", stage="do")
            else:
                task = Task("drink", life.inv.chest_x, "сундук")
        elif act == "rest":
            task = Task("rest", None, None, stage="do")
            task.posture = "lie" if self.needs.tired > LIE_FROM else "sit"
        else:
            return None
        task.started = t
        if task.stage == "do":
            self._begin(task)
        else:
            life.walk.goal = task.target
        self.task = task
        return task

    def _begin(self, task):
        task.stage = "do"
        task.left = {"eat": EAT_TIME, "drink": DRINK_TIME}.get(task.act, 0.0)
        if task.act in ("eat", "drink"):
            task.posture = "stand"
        self.life.walk.goal = None

    def stop(self):
        self.task = None
        self.life.walk.goal = None

    # --- шаг ---
    def tick(self, dt, x, t, events):
        """Шаг занятия. Возвращает True, если он сейчас занят на месте (не идёт)."""
        task = self.task
        if task is None:
            return False
        if task.stage == "go":
            d = abs((task.target - x + self.life.world.width / 2) % self.life.world.width - self.life.world.width / 2)
            if d < REACH:
                if task.item == "сундук":
                    self.refill(events)
                    task.item, task.target = "фляга", None
                self._begin(task)
            elif t - task.started > GO_MAX:
                self.finish(events, done=False)
            return False
        task.left -= dt
        if task.act == "eat" and task.left <= 0:
            self._eat(t, events)
        elif task.act == "drink" and task.left <= 0:
            if task.item == "вода":
                self.needs.thirst = 0.0                # напился вволю
                self.backpack.flask = 1.0              # и набрал флягу
            else:
                self.backpack.flask = max(0.0, self.backpack.flask - SIP)
                self.needs.thirst = max(0.0, self.needs.thirst - DRINK)
            events.append(("drank", task.item))
            self.finish(events, pleasure=0.6)
        return True

    def _eat(self, t, events):
        item = self.task.item
        if item == "рюкзак":
            self.backpack.food -= 1
            taste, what = BACKPACK_PLEASURE, "рюкзак"
        else:
            taste, e = self.food.eat(item, t)
            what = e.kind
            self.prefs.learn(self.food.feats[item], taste)
        self.needs.hunger = max(0.0, self.needs.hunger - PORTION)
        self.last_meal = (taste, what)
        events.append(("ate", what, taste))
        self.flags = {"tasty": 1.0} if taste > 0.3 else ({"not_tasty": 1.0} if taste < -0.2 else {})
        self.finish(events, pleasure=(taste + 1.0) / 2.0)

    def finish(self, events, pleasure=None, done=True):
        task = self.task
        self.stop()
        if done and pleasure is not None:
            self.life.choice.after_attempt(task.act, pleasure)
        self.life.choice.switch("explore")
        events.append(("done", task.act))

    def refill(self, events):
        if not self.backpack.full():
            self.backpack.refill()
            events.append(("refill",))
