"""Жизнь человечка без окна: мир, тело, физика, мозг и речь вместе.

tick() - один шаг физики (1/60 с); каждый второй шаг решает мозг. Возвращает события:
("phrase", текст), ("jump",), ("fall",), ("land",), ("up",).
Окно только рисует; самопроверка и кадры README гоняют то же самое.
"""
import math

import numpy as np

from body.cppn import generate_body
from body.physics import DT, Human
from brain.brain import ALIVE, FALL_PENALTY, LR, PHYSICS_PER_STEP, PLAY_LR, PLAY_NOISE
from brain.observe import observe
from brain.state import summarize
from world.generate import generate

PHRASE_PAUSE = {"rare": 14.0, "normal": 7.0, "often": 3.5}
FELL_MEMORY = 4.0          # с - столько «упал недавно» остаётся в состоянии
RECENT = 8                 # фраз, которые сеть не повторяет подряд
CANDIDATES = 3             # выборок речи на одну фразу
TEMPERATURE = 0.75
LOG_EVERY = 0.5            # с - точка графика любопытства
INTENT_K = 1.0 - math.exp(-1.0 / 30)   # намерение усредняется за секунду (мозг - 30 раз в секунду)


class Life:
    def __init__(self, models, world_seed, body_seed, width, brain, frequency="normal", seed=None):
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

    @property
    def distance(self):
        return self.human.odometer

    def state(self):
        d = self.decision
        return summarize(self.human, self.world, d.turn if d else 0.0, d.want_jump if d else 0.0,
                         self.brain.curiosity, self.fell / FELL_MEMORY)

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
        return events

    def take_reward(self):
        """Средняя награда с прошлого вызова - точка графика обучения в статистике."""
        if not self.reward_n:
            return None
        avg = self.reward_sum / self.reward_n
        self.reward_sum, self.reward_n = 0.0, 0
        return avg
