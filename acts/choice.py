"""Чем заняться: одна скука и один выбор на все занятия (спецификация 2.0, раздел 10).

У каждого занятия (гулять, гитара, потом каждая игрушка, плавать, собака, есть, пить,
отдыхать):
- **нравится** - медленное среднее удовольствия P после каждой попытки (α = 0.05);
  у «гулять» удовольствие - интерес мира: ошибка предсказателя мозга против обычной;
- **скука** - растёт, пока занятие идёт подряд (половина тяги уходит за HALF[...] минут
  игрового времени), и спадает, пока его нет (за RECOVER); плоский результат (удовольствие
  не меняется FLAT_N попыток) добавляет скуки;
- **тяга** = нравится x (1 - скука) + срочность нужды + сдвиг настройки «Как часто».

Выбор - softmax по тягам с температурой T; к тяге текущего занятия прибавляется STICK -
это гистерезис: сменить занятие он хочет, когда другое тянет заметно сильнее, а не от
каждого колебания. Ещё: не меньше MIN_ATTEMPTS попыток подряд и MIN_WORLD секунд в мире
перед сменой, если нужда не выше 0.8. Это не таймер: он встаёт, когда другое занятие тянет
сильнее. Замер без STICK (T 0.05): сеансы гитары по 3 попытки, медиана 0.5 минуты - он
садился и вставал от шума.

Замер до (1.1.0, `tools/measure_sessions.py`) - в плане: сеансы гитары не кончались,
интерес к музыке держался на новизне фраз, а скуки от длительности не было.
"""
import math

import numpy as np

HALF = {"rare": 3.5, "sometimes": 5.0, "often": 7.0}    # мин - половина тяги гитары уходит
BIAS = {"rare": -0.08, "sometimes": 0.0, "often": 0.08}  # сдвиг тяги гитары
EXPLORE_HALF = 6.0          # мин - половина тяги мира уходит, если гулять подряд
RECOVER = 3.0               # мин - за столько скука спадает в e раз, пока занятия нет
LIKE_K = 0.05               # нравится - медленное среднее удовольствия
WORLD_TAU = 60.0            # с - интерес мира - медленное среднее за минуту
FLAT_N = 8                  # попыток без изменения - скучно
FLAT_SPREAD = 0.06
FLAT_BOREDOM = 0.05
MIN_ATTEMPTS = 3
MIN_WORLD = 20.0            # с
URGENT = 0.8                # нужда выше - гистерезис не держит
T = 0.02                    # температура выбора
STICK = 0.2                 # прибавка к тяге текущего занятия - гистерезис
NEED_BASE = -0.1            # тяга есть, пить, отдыхать без нужды
NEEDS = ("eat", "drink", "rest")
DECIDE_EVERY = 5.0          # с между точками выбора, пока гуляет
MOOD_K = 1.0 - math.exp(-1.0 / 120.0)   # настроение - среднее удовольствия за ~2 минуты
REF_ERR = 0.0025            # обычная ошибка предсказателя на ходу (замер 1.1.0)


def urgent_pull():
    """Нужда URGENT в единицах срочности (acts/needs.py)."""
    from acts.needs import URGE_FROM, URGE_K
    return URGE_K * ((URGENT - URGE_FROM) / (1.0 - URGE_FROM)) ** 2


class Activity:
    def __init__(self, name, half_min, liking=0.5, bias=0.0, attempts=True, need=False, play=False):
        self.name = name
        self.half = half_min
        self.base_half = half_min    # без «Как часто»
        self.play = play             # игра (гитара, игрушка, плавать, собака): «Как часто» её касается
        self.liking = liking
        self.boredom = 0.0
        self.bias = bias
        self.attempts_based = attempts
        self.need = need             # есть, пить, отдыхать: тянет только нужда
        self.history = []            # последние удовольствия

    def rate(self):
        return math.log(2.0) / (self.half * 60.0)

    def pull(self, urgency=0.0):
        if self.need:
            # без нужды есть не тянет, как бы вкусно ни было (что вкусно - знает его вкус, а
            # не тяга занятия); NEED_BASE - чтобы слабая нужда не перебивала игру
            return urgency + NEED_BASE
        return self.liking * (1.0 - self.boredom) + self.bias + urgency


class Choice:
    def __init__(self, freq="sometimes", rng=None):
        self.rng = rng or np.random.default_rng()
        self.acts = {"explore": Activity("explore", EXPLORE_HALF, liking=0.5, attempts=False)}
        self.current = "explore"
        self.attempts = 0            # попыток подряд в текущем
        self.since = 0.0             # с в текущем
        self.mood = 0.0              # -1..1
        self.freq = freq
        self.add("guitar", HALF["sometimes"])
        for n in NEEDS:
            self.acts[n] = Activity(n, 60.0, liking=0.0, need=True)
        self.add("swim", 5.0, liking=0.45)

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        """«Как часто» - всем играм (спецификация 2.0, раздел 10): как быстро надоедает (половина
        тяги - за base_half x HALF/HALF[sometimes]) и сдвиг тяги. У гитары base_half 5 - те же
        3.5 / 5 / 7 минут, что в 1.1.0."""
        self._freq = value if value in HALF else "sometimes"
        for a in getattr(self, "acts", {}).values():
            if a.play:
                a.half = a.base_half * HALF[self._freq] / HALF["sometimes"]
                a.bias = BIAS[self._freq]

    def add(self, name, half_min, liking=0.5, bias=None):
        if name not in self.acts:
            a = Activity(name, half_min, liking, 0.0, play=True)
            self.acts[name] = a
            a.half = half_min * HALF[self.freq] / HALF["sometimes"]
            a.bias = BIAS[self.freq] if bias is None else bias
        return self.acts[name]

    # --- время и удовольствие ---
    def tick(self, dt):
        self.since += dt
        for name, a in self.acts.items():
            if name == self.current:
                a.boredom += a.rate() * (1.0 - a.boredom) * dt
            else:
                a.boredom -= a.boredom * dt / (RECOVER * 60.0)

    def world_interest(self, err_ema, dt):
        """Пока он гуляет: интерес мира (0..1) по ошибке предсказателя - удовольствие «гулять»."""
        if self.current != "explore" or err_ema is None:
            return
        obs = min(max(err_ema / REF_ERR, 0.0), 2.0) / 2.0
        e = self.acts["explore"]
        e.liking += (obs - e.liking) * (1.0 - math.exp(-dt / WORLD_TAU))
        self.mood += ((2 * obs - 1) - self.mood) * MOOD_K * 0.02

    def after_attempt(self, name, pleasure):
        """Попытка занятия name закончилась с удовольствием 0..1."""
        a = self.acts[name]
        a.liking += LIKE_K * (pleasure - a.liking)
        a.history = (a.history + [pleasure])[-FLAT_N:]
        if len(a.history) >= FLAT_N and max(a.history) - min(a.history) < FLAT_SPREAD:
            a.boredom = min(0.99, a.boredom + FLAT_BOREDOM)
        if name == self.current:
            self.attempts += 1
        self.mood += ((2 * pleasure - 1) - self.mood) * min(1.0, MOOD_K * 30)

    # --- выбор ---
    def can_switch(self, urgency=None):
        if urgency and max(urgency.values()) > urgent_pull():
            return True
        if self.current == "explore":
            return self.since >= MIN_WORLD
        return self.attempts >= MIN_ATTEMPTS

    def pulls(self, available, urgency=None):
        urgency = urgency or {}
        return {n: self.acts[n].pull(urgency.get(n, 0.0)) for n in available}

    def probabilities(self, available, urgency=None):
        p = self.pulls(available, urgency)
        names = list(p)
        z = np.array([p[n] + (STICK if n == self.current else 0.0) for n in names]) / T
        z = np.exp(z - z.max())
        return dict(zip(names, z / z.sum()))

    def choose(self, available, urgency=None):
        """Точка выбора: остаться или сменить занятие. available - что сейчас можно."""
        if self.current not in available:
            available = list(available) + [self.current]
        if not self.can_switch(urgency):
            return self.current
        probs = self.probabilities(available, urgency)
        names = list(probs)
        pick = names[int(self.rng.choice(len(names), p=[probs[n] for n in names]))]
        if pick != self.current:
            self.switch(pick)
        return pick

    def switch(self, name):
        self.current = name
        self.attempts = 0
        self.since = 0.0

    def to_dict(self):
        return {"acts": {n: {"liking": a.liking, "boredom": a.boredom} for n, a in self.acts.items()},
                "mood": self.mood}

    def load(self, d):
        if not isinstance(d, dict):
            return self
        for n, v in (d.get("acts") or {}).items():
            if n in self.acts and isinstance(v, dict):
                for k in ("liking", "boredom"):
                    x = v.get(k)
                    if isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0:
                        setattr(self.acts[n], k, float(x))
        m = d.get("mood")
        if isinstance(m, (int, float)) and not isinstance(m, bool) and -1.0 <= m <= 1.0:
            self.mood = float(m)
        return self


def pleasure(success, previous=(), owner=None):
    """Удовольствие от попытки, 0..1 (спецификация 2.0, раздел 10): успех (0..1) и рост -
    эта попытка против медианы 20 прошлых; оценка владельца (+1 / -1), если есть, - половина."""
    success = float(np.clip(success, 0.0, 1.0))
    prev = list(previous)[-20:]
    growth = 0.5 if not prev else float(np.clip(0.5 + 2.0 * (success - float(np.median(prev))), 0.0, 1.0))
    p = 0.6 * success + 0.4 * growth
    if owner is not None:
        p = 0.5 * p + 0.5 * (1.0 + float(np.clip(owner, -1, 1))) / 2.0
    return p
