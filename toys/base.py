"""Общее для игрушек: журнал попыток, вид из сети игрушек, сессия игры.

- **Журнал** - тот же, что у гитары (music/journal.py, с прореживанием), файл
  journals/<зерно тела>/<игрушка>.json; в записи - вид попытки, параметры, успех,
  удовольствие и траектория (для «Попыток»: повтор первой и последней).
- **Вид** - маленькая сеть игрушек по зерну сундука (зерно тела): цвета и пропорции.
- **Сессия** - игра с игрушкой в жизни: попытки одна за другой; после каждой - удовольствие в
  выбор занятий и решение, играть ли дальше (общий выбор, общая скука). Подкласс решает,
  что такое попытка.
"""
import colorsys
import math

import numpy as np

import paths
from music.journal import Journal
from world.cppn import ACTS, Z_DIM, seed_vector

LOOK_LAYERS = (Z_DIM + 3, 10, 4)


def _look_net():
    rng = np.random.default_rng(0x70E)
    ws, bs = [], []
    for a, b in zip(LOOK_LAYERS, LOOK_LAYERS[1:]):
        ws.append(rng.normal(0.0, 1.2 / math.sqrt(a), (a, b)))
        bs.append(rng.normal(0.0, 0.2, b))
    return ws, bs


_LOOK = _look_net()


def look(chest_seed, toy_index):
    """(основной цвет, второй цвет, пропорция 0..1) игрушки - выход сети игрушек."""
    h = np.concatenate([seed_vector(int(chest_seed) * 31 + 7), [math.sin(toy_index), math.cos(toy_index), 1.0]])[None]
    ws, bs = _LOOK
    for i, (w, b) in enumerate(zip(ws, bs)):
        h = h @ w + b
        if i < len(ws) - 1:
            out = np.empty_like(h)
            for j in range(h.shape[1]):
                out[:, j] = ACTS[j % len(ACTS)](h[:, j])
            h = out
    o = 1.0 / (1.0 + np.exp(-h[0]))
    c1 = colorsys.hsv_to_rgb(float(o[0]), 0.55 + 0.4 * o[1], 0.75 + 0.2 * o[2])
    c2 = colorsys.hsv_to_rgb(float((o[0] + 0.5) % 1.0), 0.3 + 0.4 * o[2], 0.9)
    return tuple(round(c, 4) for c in c1), tuple(round(c, 4) for c in c2), float(o[3])


def journal(body_seed, toy):
    return Journal.load(paths.user_file("journals", str(int(body_seed)), toy + ".json"))


class Session:
    """Игра с игрушкой. Подкласс задаёт start_attempt() и attempt_tick() (возвращает
    результат попытки или None, пока идёт)."""

    name = "toy"

    def __init__(self, life, head, journal_):
        self.life = life
        self.head = head
        self.journal = journal_
        self.attempts = 0
        self.scores = []
        self.active = False

    def begin(self):
        self.active = True
        self.start_attempt()

    def stop(self):
        self.active = False

    def tick(self, dt, events):
        """Шаг игры. True - он занят игрушкой."""
        if not self.active:
            return False
        result = self.attempt_tick(dt, events)
        if result is None:
            return True
        success, entry = result
        self.attempts += 1
        from acts.choice import pleasure
        p = pleasure(success, self.scores)
        self.scores = (self.scores + [success])[-20:]
        entry.update({"success": round(float(success), 4), "pleasure": round(float(p), 4)})
        saved = self.journal.add(entry)
        events.append(("toy_attempt", self.name, saved))
        life = self.life
        life.choice.after_attempt(self.name, p)
        if life.choice.choose(life.available(), life.needs.urgency()) != self.name:
            self.active = False
            events.append(("toy_done", self.name))
            return False
        if self.ready():
            self.start_attempt()
        return True

    def ready(self):
        """Можно сразу следующую попытку (игрушка в руках)."""
        return True

    # для подклассов
    def start_attempt(self):
        raise NotImplementedError

    def attempt_tick(self, dt, events):
        raise NotImplementedError
