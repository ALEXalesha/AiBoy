"""Жизнь человечка без окна: мир, тело, физика, мозг, речь и гитара вместе.

tick() - один шаг физики (1/60 с); каждый второй шаг решает мозг. Возвращает события:
("phrase", текст), ("jump",), ("fall",), ("land",), ("up",), и гитарные: ("pickup",) -
подобрал гитару, ("sit",) - сел играть, ("attempt_start", попытка), ("attempt", запись
журнала), ("stand",) - встал.

Гитара лежит в полутора метрах перед ним, подбирает, проходя рядом, и дальше носит её с
собой. Играть ли - решает music/mood.py: мир надоел сильнее музыки - садится. Пока играет,
мозг ходьбы не делает шагов и не учится, тело сидит, руки - по попытке.
Окно только рисует и играет звук; самопроверка и кадры README гоняют то же самое.
"""
import math

import numpy as np

from body.cppn import generate_body
from body.physics import DT, Human
from brain.brain import ALIVE, FALL_PENALTY, LR, PHYSICS_PER_STEP, PLAY_LR, PLAY_NOISE
from brain.observe import observe
from brain.state import summarize
from music import hands
from music.mood import Mood
from music.session import FPS
from world.generate import generate

PHRASE_PAUSE = {"rare": 14.0, "normal": 7.0, "often": 3.5}
FELL_MEMORY = 4.0          # с - столько «упал недавно» остаётся в состоянии
RECENT = 8                 # фраз, которые сеть не повторяет подряд
CANDIDATES = 3             # выборок речи на одну фразу
TEMPERATURE = 0.75
LOG_EVERY = 0.5            # с - точка графика любопытства
INTENT_K = 1.0 - math.exp(-1.0 / 30)   # намерение усредняется за секунду (мозг - 30 раз в секунду)
GUITAR_AHEAD = 1.5         # м - где лежит гитара от места появления
PICKUP = 0.8               # м - подбирает, проходя ближе
ATTEMPT_PAUSE = 0.8        # с тишины между попытками
SIT_TIME = 1.0             # с - садится и берёт гитару, прежде чем заиграть


class Life:
    def __init__(self, models, world_seed, body_seed, width, brain, frequency="normal", seed=None,
                 guitarist=None, has_guitar=False, guitar_freq="sometimes", mood=None):
        self.models = models
        self.world_seed, self.body_seed = int(world_seed), int(body_seed)
        self.world = generate(models.world_genome, world_seed, width)
        self.body = generate_body(models.body_genome, body_seed)
        self.human = Human(self.body, self.world, x=self.world.width * 0.5)
        self.brain = brain
        self.brain.reset_memory()
        self.brain.opt.lr = LR * PLAY_LR
        self.rng = np.random.default_rng(seed if seed is not None else [world_seed, body_seed])
        self.frequency = frequency
        self.learn = True
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
        self.has_guitar = has_guitar
        self.guitar_x = self.human.px + GUITAR_AHEAD * self.human.facing
        self.mood = mood if mood is not None else Mood(guitar_freq)
        self.playing = False
        self.sitting_for = 0.0
        self.attempt = None
        self.play_t = 0.0
        self.pending_rating = None
        self.last_entry = None
        self.guitar_time = 0.0
        self.music_flags = {}

    @property
    def distance(self):
        return self.human.odometer

    def state(self):
        d = self.decision
        return summarize(self.human, self.world, d.turn if d else 0.0, d.want_jump if d else 0.0,
                         self.brain.curiosity, self.fell / FELL_MEMORY, music=self.music_flags)

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

    def tick(self):
        if self.playing:
            return self._guitar_tick()
        events = []
        if self.k % PHYSICS_PER_STEP == 0:
            obs = observe(self.human, self.world)
            self.last_obs, self.obs_x = obs, self.human.px
            self.obs_ground = self.world.height_at(self.human.px)
            self.decision, reward = self.brain.step(obs, self.extra, learn=self.learn, noise=self.noise)
            self.intent += (self.decision.turn - self.intent) * INTENT_K
            if reward is not None:
                self.reward_sum += reward
                self.reward_n += 1
            d = self.decision
            if self.since_phrase >= PHRASE_PAUSE[self.frequency] and self.rng.random() < d.say:
                events.append(("phrase", self.say()))
            self.extra = 0.0
        d = self.decision
        happened = self.human.step(d.targets, d.turn, d.jump and self.k % PHYSICS_PER_STEP == 0)
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

    # --- гитара ---
    def _world_guitar(self, events):
        self.mood.world_tick(self.brain.err_ema, DT)
        if not self.has_guitar:
            w = self.world.width
            dx = (self.guitar_x - self.human.px + w / 2) % w - w / 2
            if abs(dx) < PICKUP:
                self.has_guitar = True
                events.append(("pickup",))
        elif (self.guitarist is not None and self.mood.want_play() and self.human.grounded
              and self.human.fallen <= 0):
            self.sit_down()
            events.append(("sit",))

    def sit_down(self):
        self.playing = True
        self.mood.start()
        self.attempt = None
        self.sitting_for = 0.0
        self.music_flags = {"playing": 1.0}

    def stand_up(self):
        self.playing = False
        self.attempt = None
        self.mood.stop()
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
            self.mood.after_attempt(novelty, progress)
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
            if self.mood.want_stop():
                self.stand_up()
                events.append(("stand",))
        self._clock_tick()
        return events

    def take_reward(self):
        """Средняя награда с прошлого вызова - точка графика обучения в статистике."""
        if not self.reward_n:
            return None
        avg = self.reward_sum / self.reward_n
        self.reward_sum, self.reward_n = 0.0, 0
        return avg
