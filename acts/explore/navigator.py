"""Навигатор - голова занятия «гулять»: куда идти.

Раз в NAV_EVERY секунд, пока он идёт по земле, сеть по памяти мест (brain/places.py)
выбирает: идти дальше, повернуть назад или остановиться и осмотреться. Вход - сводка
«впереди» и «сзади» на трёх расстояниях (новизна, интерес), тяга меток (гитара и т. п.)
в обе стороны, доля изученного мира, сколько прошло с разворота, интерес здесь.

Учится сама, актёр-критик (acts/policy.py): награда решения - то, что случилось до
следующего решения, плюс оценка критиком состояния, в которое он пришёл (горизонт около
10 с при GAMMA 0.9 на решение):
- новизна каждой новой клетки, куда он пришёл: 1 / (1 + сколько раз уже был);
- запомненный интерес этой клетки минус обычный (бугры, деревья - плюс, ровное - ноль);
- тяга метки, куда тянет (гитара, когда хочется играть): APPROACH за каждый метр ближе к
  ней и 5 - когда дошёл (только за приход сигнал слишком редкий: замер - за 5 минут он
  доходил до метки 0.3 раза в среднем);
- минус TURN_COST за разворот (развернуться - это усилие и время);
- «осмотреться» приносит интерес того, что рядом и увидено впервые, - рассмотреть то,
  что только что прошёл.
Это не правило «когда поворачивать»: в новом мире новизна впереди выше, и сеть сама
выучивает идти вперёд; в изученном новизна везде почти одна, и решает запомненный
интерес и тяга - иногда назад. Гистерезис против «туда-сюда»: после разворота COMMIT
секунд назад нельзя (как «не меньше 3 попыток» у выбора занятий).
"""
import numpy as np

from acts.policy import ChoiceActorCritic
from brain import places

FORWARD, BACK, LOOK = 0, 1, 2
ACTIONS = ("вперёд", "назад", "осмотреться")
IN = places.N_FEATURES + 2
NAV_EVERY = 1.0          # с между решениями
COMMIT = 12.0           # с после разворота - назад нельзя
STUCK = 8.0             # с без продвижения к цели - застрял
LOOK_GAP = 15.0          # с между «осмотреться»
LOOK_TIME = 2.5          # с стоит и смотрит
NOVEL = 1.0
INTEREST = 0.5
TURN_COST = 2.5
LOOK_K = 0.5
REACH = 1.0              # м - дошёл до метки
APPROACH = 0.3           # за метр ближе к метке, куда тянет


class Navigator(ChoiceActorCritic):
    def __init__(self, seed=0):
        super().__init__(IN, len(ACTIONS), hidden=24, seed=seed, lr=3e-3, entropy=0.1, gamma=0.9)


class Walk:
    """Состояние навигатора в одной жизни: память мест, что решено, что ещё учится."""

    def __init__(self, width, facing, navigator, rng=None):
        self.places = places.Places(width)
        self.nav = navigator
        self.rng = rng or np.random.default_rng()
        self.want = facing           # куда он хочет смотреть и идти
        self.since_turn = COMMIT     # в начале развернуться можно
        self.since_look = LOOK_GAP
        self.clock = NAV_EVERY
        self.look_left = 0.0
        self.prev = None             # [вход, маска, действие, награда до следующего решения, когда]
        self.turns = 0               # разворотов по своей воле
        self.looks = 0
        self.last_action = FORWARD
        self.pulls = {}              # метка -> насколько тянет туда сейчас (0..1)
        self.reached = set()
        self.learn = True
        self.mean = 1.0              # средний интерес знакомых клеток (для «интерес здесь»)
        self.gap = {}                # метка -> расстояние до неё на прошлом шаге
        self._goal = None            # x, куда он идёт за чем-то (еда, сундук, гитара) - кратчайшим путём
        self.goal_best = None        # (ближе всего к цели за этот заход, когда)
        self.goal_way = 0            # 0 - кратчайшим путём; ±1 - застрял, идёт в обход по кольцу

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, x):
        self._goal = None if x is None else float(x)
        self.goal_best, self.goal_way = None, 0
        self.goal_fixed = False      # сторона задана (гора: лёгкий путь - только с одной стороны) - не обходить

    @property
    def looking(self):
        return self.look_left > 0

    def features(self, x, facing):
        f = self.places.features(x, facing, self.pulls)
        inter = self.places.interest()
        here = inter[self.places.index(x)]
        return np.concatenate([f, [min(self.since_turn / 20.0, 1.0), float(np.clip(here - 1.0, -1, 2))]]
                              ).astype(np.float64)

    def mask(self):
        return np.array([True, self.since_turn >= COMMIT, self.since_look >= LOOK_GAP])

    def _look_value(self, x):
        """Интерес того, что рядом (5 м в обе стороны) и увидено впервые-вторые."""
        inter = self.places.interest()
        best = 0.0
        for d in np.arange(-5.0, 5.01, 1.0):
            b = self.places.index(x + d)
            if 0 < self.places.visits[b] <= 2:
                best = max(best, inter[b] - 1.0)
        return LOOK_K * best

    def tick(self, t, dt, x, facing, interest, can_decide):
        """Шаг жизни: память мест, награды идущих решений, новое решение раз в NAV_EVERY.
        Возвращает, куда он хочет смотреть (±1)."""
        self.since_turn += dt
        self.since_look += dt
        self.look_left = max(0.0, self.look_left - dt)
        entered, before = self.places.visit(x, t, interest)
        r = 0.0
        if entered:
            b = self.places.index(x)
            rel = self.places.isum[b] / max(self.places.icount[b], 1) / (self.mean + 1e-9)
            r += NOVEL / (1.0 + before) + INTEREST * float(np.clip(rel - 1.0, -1.0, 2.0))
        for name, want in self.pulls.items():
            mx = self.places.marks.get(name)
            if mx is None or want <= 0:
                self.gap.pop(name, None)
                continue
            d = abs((mx - x + self.places.width / 2) % self.places.width - self.places.width / 2)
            if name in self.gap:
                r += APPROACH * want * (self.gap[name] - d)
            self.gap[name] = d
            near = d < REACH
            if near and name not in self.reached:
                self.reached.add(name)
                r += want * 5.0
            elif not near:
                self.reached.discard(name)
        if r and self.prev is not None:
            self.prev[3] += r
        self.clock += dt
        if self.goal is not None:
            # идёт за чем-то конкретным: кратчайшим путём по кольцу, навигатор не спрашивается
            # (замер: по одной тяге метки он за 70 с так и не дошёл до сундука с водой). Застрял
            # (STUCK с ни на метр ближе - во впадине, ногами не вылезти) - идёт в обход: мир
            # кольцо, другой дорогой тоже дойдёт
            right = self.places.distance(x, self.goal, 1)
            left = self.places.width - right
            way = self.goal_way or (1 if right <= left else -1)
            # продвижение - по той дороге, которой идёт: в обход кратчайшее расстояние сначала
            # растёт (замер: через 8 с обход считался «застрял», и он ходил туда-сюда 70 с)
            d = right if way > 0 else left
            if self.goal_best is None or d < self.goal_best[0] - 1.0:
                self.goal_best = (d, t)
            elif t - self.goal_best[1] > STUCK and not getattr(self, "goal_fixed", False):
                self.goal_way = -way
                self.goal_best = (1e9, t)            # заново, по новой дороге
                way = self.goal_way
            self.want = way
            self.look_left = 0.0
            self.prev = None
            return self.want
        if can_decide and not self.looking and self.clock >= NAV_EVERY:
            self.clock = 0.0
            self.decide(t, x, facing)
        return self.want

    def decide(self, t, x, facing):
        known = self.places.icount > 0
        if known.any():
            self.mean = float((self.places.isum[known] / self.places.icount[known]).mean())
        feats = self.features(x, facing)
        if self.prev is not None and self.learn:
            f0, m0, a0, r0, t0 = self.prev
            self.nav.learn_step(f0, m0, a0, r0, feats, steps=max(1.0, (t - t0) / NAV_EVERY))
        mask = self.mask()
        a, _ = self.nav.choose(feats, mask, rng=self.rng)
        direct = 0.0
        if a == BACK:
            self.want = -facing
            self.since_turn = 0.0
            self.turns += 1
            direct -= TURN_COST
        elif a == LOOK:
            self.look_left = LOOK_TIME
            self.since_look = 0.0
            self.looks += 1
            direct += self._look_value(x)
        else:
            self.want = facing
        self.last_action = a
        self.prev = [feats, mask, a, direct, t]
        return a

    def rest(self, facing):
        """Сел играть, сменилось занятие: череда решений обрывается без обучения."""
        self.prev = None
        self.look_left = 0.0
        self.want = facing
        self.clock = NAV_EVERY
