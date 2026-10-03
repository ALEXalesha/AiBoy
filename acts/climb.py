"""Занятие «гора» (спецификация 2.0, раздел 8a): заметил гору - может пройти мимо, подняться лёгким
путём или полезть на стену (руками или со снаряжением). Залез - флаг (в первый раз) и радость.

**Путь** решает softmax по ценностям, не правило:
- лёгкий путь: тем ценнее, чем больше страх (после срывов) и усталость;
- стена руками: насколько ему кажется, что получится (навык skills/climb.Climber), плюс новизна
  непокорённой горы, минус страх и усталость;
- стена с верёвкой / ледорубом: то же по своему навыку снаряжения, если оно есть в сундуке.

**Лезет** - режим, как вода (acts/swim.py): тело на стене ставит сам (руки к зацепам, ноги
упираются), ход раз в MOVE_T. Срыв - съезжает вниз до полки или крюка (не падает): «ой», немного
устал, страх растёт. Застрял (STUCK с без высоты) - слезает. Четыре срыва за попытку - хватит.

**На вершине** (любым путём): в первый раз втыкает флаг (он остаётся в мире), потом радуется -
руки вверх и подпрыгивает, фраза, звук, настроение вверх, вершина - в память мест.
"""
import math

import numpy as np

from skills import climb as sk
from world import mountains as mt

MOVE_T = 0.7                # с - ход
SLIDE_V = 3.0               # м/с - съезжает со стены
STUCK = 25.0                # с без высоты - слезает
CELEBRATE = 2.6             # с радуется
NOTICE = 300.0              # м - замечает гору (60-100 м высоты видно издалека)
FEAR_FALL = 0.15            # страха за срыв
FEAR_TOP = 0.2              # страха долой за вершину
FEAR_DECAY = 1.0 / 900.0    # страх спадает сам, в секунду
ROUTE_T = 0.1
CONQUERED = 0.3             # покорённая гора тянет во столько раз слабее новой
WAY_COST = 400.0            # м дороги - минус единица ценности пути (замер: шёл к лёгкому за 700 м, а стена - в 120)
CLIMB_POSE = (0.05, 0.05, 2.7, 2.3, 0.6, 0.9, 0.9, 0.3, 1.2, 0.6)
ARMS_UP = (0.05, 0.05, 2.9, 2.9, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1)


def notice(peaks, x, width):
    """Ближайшая замеченная гора (по ближнему краю её следа) или None."""
    best = None
    for k, m in enumerate(peaks):
        s = float(mt.local(m, x, width))
        lo, hi = -(mt.TOP / 2 + m.easy_len), mt.TOP / 2 + m.face_len
        out = max(0.0, lo - s, s - hi)
        if out <= NOTICE and (best is None or out < best[1]):
            best = (k, out)
    return None if best is None else best[0]


class Climb:
    def __init__(self, life, climber, rope=None, axe=None):
        self.life = life
        self.climber = climber
        self.rope_skill = rope
        self.axe_skill = axe
        self.fear = 0.0
        self.mode = None            # None | "face" | "slide" | "celebrate"
        self.route = None           # "easy" | "face" | "rope" | "axe"
        self.target = None          # индекс горы
        self.walls = {}
        self.y = 0.0
        self.fatigue = 0.0
        self.anchor = None
        self.falls = 0
        self.best = 0.0
        self.best_t = 0.0
        self.t = 0.0
        self.move_t = 0.0
        self.slide_to = 0.0
        self.celebrate_t = 0.0
        self.moves = 0
        self.flags = []             # x вершин с флагами (мир)
        self.conquered = set()      # индексы покорённых гор этого мира
        self.summits = 0            # сколько раз был на вершине
        self.ending = False
        self.go = None              # (гора, путь) - идёт к ней
        self.go_max = 600.0         # с - столько идёт, потом решает заново
        self.on_top = None          # на вершине какой горы стоит сейчас (чтобы радоваться раз)

    # --- что выбрать ---
    def peaks(self):
        return getattr(self.life.world, "mountains", None) or []

    def available(self, x):
        return notice(self.peaks(), x, self.life.world.width) is not None

    def novelty(self, k):
        return CONQUERED if k in self.conquered else 1.0

    def bias(self, x):
        """Прибавка к тяге «гора» сейчас: новизна, страх."""
        k = notice(self.peaks(), x, self.life.world.width)
        if k is None:
            return 0.0
        return 0.15 * self.novelty(k) - 0.15 * (1.0 - self.novelty(k)) - 0.4 * self.fear

    def routes(self, x, tools, tired):
        """{путь: вероятность}. tools - какое снаряжение можно взять (имена реестра)."""
        k = notice(self.peaks(), x, self.life.world.width)
        if k is None:
            return {}, None
        nov = self.novelty(k)
        d_easy, d_face = self.approach(k, x)
        v = {"easy": 0.45 + 0.5 * self.fear + 0.5 * tired - d_easy / WAY_COST,
             "face": self.climber.confidence() + 0.25 * nov - 0.6 * self.fear - 0.5 * tired - d_face / WAY_COST}
        for name, skill in (("rope", self.rope_skill), ("axe", self.axe_skill)):
            if name in tools:
                conf = float(np.mean(skill.history[-10:])) if getattr(skill, "history", None) else 0.35
                v[name] = conf + 0.3 * nov - 0.5 * self.fear - 0.5 * tired - 0.05 - d_face / WAY_COST
        names = list(v)
        z = np.array([v[n] for n in names]) / ROUTE_T
        z = np.exp(z - z.max())
        return dict(zip(names, z / z.sum())), k

    def approach(self, k, x):
        """Сколько идти (м) до начала лёгкого пути и до подножия стены. Мир - кольцо, но путь
        к каждому - со своей стороны: к лёгкому - по ходу +steep (через стену не пройти), к
        стене - по ходу -steep. На самом лёгком пути - 0."""
        m = self.peaks()[k]
        w = self.life.world.width
        s = float(mt.local(m, x, w))
        if -(mt.TOP / 2 + m.easy_len) <= s <= mt.TOP / 2:
            d_easy = 0.0
        else:
            d_easy = ((m.easy_foot() - x) * m.steep) % w
        d_face = ((m.face_foot() - x) * -m.steep) % w
        return d_easy, d_face

    # --- стена ---
    def wall(self, k):
        if k not in self.walls:
            m = self.peaks()[k]
            self.walls[k] = sk.Wall(self.life.world_seed, k, mt.face_wall(self.life.world, m)[0] or m.height)
        return self.walls[k]

    def face_point(self, k, y):
        """Где он на стене на высоте y: (x, y таза)."""
        m = self.peaks()[k]
        foot = m.face_foot()
        g = self.life.world.height_at(foot)
        w = self.wall(k)
        frac = min(1.0, y / max(w.height, 1e-6))
        x = foot - m.steep * (frac * (abs(foot - m.summit) - mt.TOP / 2))
        return x - m.steep * 0.25, g + y + 0.6

    def start_face(self, k, route):
        self.mode, self.target, self.route = "face", k, route
        self.y, self.fatigue, self.anchor, self.falls = 0.0, 0.0, None, 0
        self.best, self.best_t, self.t, self.move_t = 0.0, 0.0, 0.0, 0.0
        self.moves = 0
        self._place()

    def _place(self, pose=CLIMB_POSE):
        h = self.life.human
        x, y = self.face_point(self.target, self.y)
        h.px, h.py = x, y
        h.vx = h.vy = 0.0
        m = self.peaks()[self.target]
        h.facing = -m.steep                       # лицом к стене
        a = list(pose)
        if pose is CLIMB_POSE and self.moves % 2:
            a[2], a[3] = a[3], a[2]                # руки по очереди
            a[6], a[7] = a[7], a[6]
        h.angles = np.array(a, dtype=float)
        h.grounded = False

    def tick(self, dt, events):
        """Шаг режима. True - занят (тело ставит он)."""
        if self.mode is None:
            return False
        self.fear = max(0.0, self.fear - FEAR_DECAY * dt)
        if self.mode == "celebrate":
            self.celebrate_t -= dt
            h = self.life.human
            h.angles = np.array(ARMS_UP, dtype=float)
            if CELEBRATE - self.celebrate_t < 0.5:
                h.py += 1.5 * dt                    # подпрыгивает
            elif h.py > self.life.world.height_at(h.px) + h.body.leg():
                h.py = max(self.life.world.height_at(h.px) + h.body.leg(), h.py - 1.5 * dt)
            if self.celebrate_t <= 0:
                self.mode = None
                h.py = self.life.world.height_at(h.px) + h.body.leg()
                events.append(("celebrated",))
            return True
        if self.mode == "slide":
            self.y = max(self.slide_to, self.y - SLIDE_V * dt)
            self._place()
            if self.y <= self.slide_to + 1e-6:
                if self.falls >= sk.FALLS or self.slide_to <= 0.0 and self.ending:
                    self._finish(events, top=False)
                else:
                    self.mode = "face"
            return True
        # на стене
        self.t += dt
        self.move_t += dt
        if self.move_t >= MOVE_T:
            self.move_t = 0.0
            self._move(events)
        if self.mode == "face" and self.t - self.best_t > STUCK:
            self._slide(0.0, events, ending=True)     # застрял - слезает
        if self.mode == "face":
            self._place()
        return True

    def _move(self, events):
        w = self.wall(self.target)
        life = self.life
        arm = life.body.upper_arm + life.body.forearm
        if self.route == "rope" and self.rope_skill is not None and (self.anchor is None or self.y >= self.anchor):
            hooked = self.rope_skill.throw(w, self.y, life.rng, learn=life.learn)
            events.append(("sound", "throw", 0.5, 0.05))
            if hooked is not None:
                self.anchor = hooked
                events.append(("sound", "clink", 0.6, 0.05))
        axe = self.axe_skill if self.route == "axe" else None
        if axe is not None:
            events.append(("sound", "strike", 0.6, 0.05))
        rope = self.anchor is not None and self.anchor > self.y
        ny, self.fatigue, _ = sk.move(self.climber, w, self.y, arm, self.fatigue, rope, life.rng, axe, learn=life.learn)
        self.moves += 1
        if ny is None:
            self.falls += 1
            stop = w.ledge_below(self.y)
            if self.anchor is not None and self.anchor <= self.y:
                stop = max(stop, self.anchor)
            self._slide(stop, events, ending=self.falls >= sk.FALLS)
            return
        self.y = ny
        events.append(("sound", "scrape", 0.3, 0.05))
        if w.ledge[w.near(ny)]:
            self.fatigue = 0.0
        if self.y > self.best + 0.05:
            self.best, self.best_t = self.y, self.t
        if self.y >= w.height - sk.HOLD_STEP / 2:
            self._top(events)

    def _slide(self, stop, events, ending=False):
        """Срыв или спуск: съезжает до stop - без вреда. «Ой», немного устал, страх."""
        self.mode, self.slide_to, self.ending = "slide", stop, ending
        life = self.life
        life.needs.tired = min(1.0, life.needs.tired + 0.03)
        self.fear = min(1.0, self.fear + FEAR_FALL)
        self.fatigue = 0.0 if stop in self.wall(self.target).ys[self.wall(self.target).ledge] or stop == 0.0 else self.fatigue
        events.append(("climb_fall", round(self.y, 2), round(stop, 2)))
        life.climb_flags = {"climb_fell": 1.0}
        events.append(("phrase", life.say()))
        life.climb_flags = {}

    def _top(self, events):
        m = self.peaks()[self.target]
        h = self.life.human
        h.px = m.summit + m.steep * (mt.TOP / 2 - 0.4)
        h.py = self.life.world.height_at(h.px) + h.body.leg()
        h.grounded = True
        self._finish(events, top=True)

    def _finish(self, events, top):
        share = 1.0 if top else min(1.0, self.best / max(1e-6, self.wall(self.target).height))
        c = self.climber
        c.attempts += 1
        c.history = (c.history + [share])[-50:]
        if top:
            c.tops += 1
        tool = {"rope": self.rope_skill, "axe": self.axe_skill}.get(self.route)
        if tool is not None:
            tool.history = (getattr(tool, "history", []) + [share])[-50:]
        events.append(("climb_attempt", {"route": self.route, "height": round(self.best, 2), "share": round(share, 3),
                                         "falls": self.falls, "top": top}))
        self.mode = None
        if top:
            self.summit(self.target, events)
        else:
            self.life.after_climb(share)

    # --- вершина ---
    def summit(self, k, events):
        """На вершине горы k: флаг (в первый раз), радость."""
        m = self.peaks()[k]
        first = k not in self.conquered
        self.conquered.add(k)
        self.summits += 1
        if not any(abs(((f - m.summit + self.life.world.width / 2) % self.life.world.width) - self.life.world.width / 2) < 1.0
                   for f in self.flags):
            self.flags.append(float(m.summit % self.life.world.width))
            events.append(("flag", float(m.summit)))
            events.append(("sound", "flag", 0.6, 0.1))
        self.fear = max(0.0, self.fear - FEAR_TOP)
        self.mode, self.celebrate_t, self.target = "celebrate", CELEBRATE, k
        events.append(("summit", k, first))
        events.append(("sound", "fanfare", 0.8, 0.1))
        self.life.on_summit(k, first)

    def to_dict(self):
        return {"fear": self.fear}

    def load(self, d):
        if isinstance(d, dict):
            v = d.get("fear")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                self.fear = float(v)
        return self
