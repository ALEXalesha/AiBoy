"""Навык «плыть» (swim): голова-сеть решает, как грести в эту попытку - сила рук, частота,
ноги, наклон тела; тело в воде (body/swim.py) плывёт или барахтается. Учится REINFORCE с
гауссовым шумом (skills/base.GaussianSkill), как все навыки: награда попытки -
проплыл (м) + голова над водой (доля времени) - усилие.

Вход: глубина (м), усталость (0..1), сколько уже попыток (доля от 200 - опыт).
Пловцом он считается, когда за последние 10 попыток в медиане плывёт быстрее SWIM_SPEED и
держит голову над водой дольше HEAD_SHARE: тогда глубокая вода ему открыта.
"""
import numpy as np

from body.swim import Stroke, SwimBody
from skills.base import GaussianSkill

ATTEMPT = 6.0            # с одна попытка
SWIMMER = 2.0            # награда, от которой попытка - удовольствие
SWIM_SPEED = 0.6         # м/с
HEAD_SHARE = 0.8
LO = [0.0, 0.4, 0.0, 0.0]
HI = [1.0, 2.5, 1.0, 1.3]
NEWBIE = [0.95, 2.3, 0.1, 1.2]   # рука, частота, ноги, наклон


def reward(dist, head_share, effort):
    return dist + 2.0 * head_share - 0.05 * effort


class Swimmer(GaussianSkill):
    def __init__(self, seed=0):
        super().__init__(3, LO, HI, hidden=24, seed=seed, sigma=0.25, lr=5e-3)
        # новичок в воде стоит стоймя и молотит руками: так он и начинает, а дальше учится
        unit = self.bounds.to_unit(np.array(NEWBIE, np.float32))
        self.net.items[-1].b[:] = np.arctanh(np.clip(unit, -0.95, 0.95))
        self.history = []
        self.skill = []              # (скорость, доля головы над водой) последних попыток

    def plan(self, depth, tired):
        x = np.array([min(depth, 3.0) / 3.0, tired, min(self.attempts / 200.0, 1.0)], np.float32)
        cmd, rec = self.act(x)
        return Stroke(*[float(v) for v in cmd]), rec

    def finish(self, rec, dist, head_share, effort, seconds=ATTEMPT):
        r = reward(dist, head_share, effort)
        self.learn(rec, r)
        self.history = (self.history + [r])[-20:]
        self.skill = (self.skill + [(dist / max(seconds, 0.5), head_share)])[-10:]
        return r

    def save(self, path):
        from nn import io
        arrays = io.collect(self.nets())
        arrays["meta.numbers"] = np.array([self.baseline, self.spread, self.attempts], np.float64)
        arrays["meta.skill"] = np.array(self.skill or [(0.0, 0.0)], np.float64)
        arrays["meta.history"] = np.array(self.history or [0.0], np.float64)
        io.save(path, arrays)

    def load(self, path):
        from nn import io
        arrays = io.load(path)
        io.restore(self.nets(), arrays)
        b, s, n = arrays["meta.numbers"]
        self.baseline, self.spread, self.attempts = float(b), float(s), int(n)
        self.skill = [tuple(map(float, r)) for r in arrays["meta.skill"]] if n else []
        self.history = [float(v) for v in arrays["meta.history"]] if n else []
        return self

    def can_swim(self):
        if len(self.skill) < 5:
            return False
        speed = float(np.median([s for s, _ in self.skill]))
        head = float(np.median([h for _, h in self.skill]))
        return speed >= SWIM_SPEED and head >= HEAD_SHARE


def try_swim(swimmer, depth=2.0, tired=0.0, seconds=ATTEMPT):
    """Одна попытка в открытой воде (для замеров и тестов): (награда, метров, доля головы над водой)."""
    stroke, rec = swimmer.plan(depth, tired)
    body = SwimBody(0.0, 0.0)
    above = 0
    n = int(seconds * 60)
    for _ in range(n):
        if body.step(stroke) > 0:
            above += 1
    r = swimmer.finish(rec, body.path, above / n, body.effort, seconds)
    return r, body.path, above / n
