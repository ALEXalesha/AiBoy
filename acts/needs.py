"""Нужды: голод, жажда, усталость (каждая 0..1) и рюкзак.

Нужды - не правила, а срочность в выборе занятий (acts/choice.py): чем сильнее нужда, тем
сильнее тянет есть, пить или отдыхать. Растут:
- голод - медленно (от 0 до 1 за HUNGER_TIME);
- жажда - быстрее, днём ещё быстрее (солнце), и от бега и плавания;
- усталость - от пройденного пути и подъёмов (и плавания, велосипеда - в своих задачах).
Отдых: сидя усталость уходит, лёжа - быстрее всего.

Рюкзак: запас еды (порции), фляга (0..1), лакомства для собаки; пополняется у сундука.

Закон: ни одна нужда не держится выше 0.9 дольше 60 с игрового времени (замер в плане).
"""
import numpy as np

HUNGER_TIME = 900.0          # с - от сытого до голодного
THIRST_TIME = 600.0          # с - от напившегося до жажды
SUN = 1.3                    # днём жажда растёт быстрее
RUN_THIRST = 0.0006          # за метр бега (быстрее 1.5 м/с)
WALK_TIRE = 0.00022          # за метр пути
CLIMB_TIRE = 0.0012          # за метр подъёма
SIT_REST = 1.0 / 240.0       # в секунду сидя
LIE_REST = 1.0 / 90.0        # в секунду лёжа
URGE_FROM = 0.3              # ниже - не срочно
URGE_K = 1.6                 # срочность при нужде 1 (замер: при 0.9 голод 0.9 не перетягивал
                             # «гулять» с его тягой до 1 - голодал по минуте и больше)
SLOW_FROM = 0.5              # усталость выше - шаг короче
SLOW_MIN = 0.75              # короче не бывает

FOOD, FLASK, TREATS = 4, 1.0, 5      # полный рюкзак


class Needs:
    def __init__(self):
        self.hunger = 0.0
        self.thirst = 0.0
        self.tired = 0.0

    def tick(self, dt, walked=0.0, climbed=0.0, speed=0.0, night=False, posture="stand"):
        self.hunger = min(1.0, self.hunger + dt / HUNGER_TIME)
        sun = 1.0 if night else SUN
        run = walked if speed > 1.5 else 0.0
        self.thirst = min(1.0, self.thirst + sun * dt / THIRST_TIME + RUN_THIRST * run)
        self.tired = min(1.0, self.tired + WALK_TIRE * walked + CLIMB_TIRE * max(0.0, climbed))
        if posture == "sit":
            self.tired = max(0.0, self.tired - SIT_REST * dt)
        elif posture == "lie":
            self.tired = max(0.0, self.tired - LIE_REST * dt)

    def urgency(self):
        """Срочность для выбора: есть, пить, отдыхать. Растёт с квадратом нужды выше URGE_FROM."""
        def u(v):
            return URGE_K * float(np.clip((v - URGE_FROM) / (1.0 - URGE_FROM), 0.0, 1.0)) ** 2
        return {"eat": u(self.hunger), "drink": u(self.thirst), "rest": u(self.tired)}

    def stride(self):
        """Во сколько короче шаг от усталости (1 - как обычно)."""
        over = max(0.0, self.tired - SLOW_FROM) / (1.0 - SLOW_FROM)
        return 1.0 - (1.0 - SLOW_MIN) * over

    def worst(self):
        return max(self.hunger, self.thirst, self.tired)

    def to_dict(self):
        return {"hunger": self.hunger, "thirst": self.thirst, "tired": self.tired}

    def load(self, d):
        if isinstance(d, dict):
            for k in ("hunger", "thirst", "tired"):
                v = d.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                    setattr(self, k, float(v))
        return self


class Backpack:
    def __init__(self):
        self.refill()

    def refill(self):
        self.food, self.flask, self.treats = FOOD, FLASK, TREATS

    def full(self):
        return self.food == FOOD and self.flask >= FLASK - 1e-9

    def to_dict(self):
        return {"food": self.food, "flask": self.flask, "treats": self.treats}

    def load(self, d):
        if isinstance(d, dict):
            f, fl, t = d.get("food"), d.get("flask"), d.get("treats")
            if isinstance(f, int) and 0 <= f <= FOOD:
                self.food = f
            if isinstance(fl, (int, float)) and not isinstance(fl, bool) and 0.0 <= fl <= FLASK:
                self.flask = float(fl)
            if isinstance(t, int) and 0 <= t <= TREATS:
                self.treats = t
        return self
