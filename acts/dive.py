"""Маска для ныряния (спецификация 2.0, раздел 5): игрушка-снаряжение из сундука. Хочет долго в
воде - смотреть рыб, дно, нырять - берёт маску. С ней дышит под водой: предел под водой - не 2.5 с
(acts/swim.UNDER), а его воздух (AIR с, у поверхности набирается снова); воздух кончился или
устал - всё та же безопасность: на берег за 5 с, не тонет.

Голова маски (skills/base.GaussianSkill) перед каждым нырком по глубине пруда, глубине рыбы,
которую видит сверху, оставшемуся воздуху и усталости решает, на какую глубину нырнуть и сколько
там держаться. Рыба мимо проплывает не сразу: держаться дольше - вернее увидеть, но больше
воздуха. Награда - увидел рыбу близко (не дальше SEE_FISH), минус - если воздух кончился раньше,
чем всплыл. Новичок ныряет как придётся и всплывает пустым; с нырками - смотрит рыб.
"""
import math

import numpy as np

from skills.base import GaussianSkill

AIR = 25.0                  # с под водой с маской
REFILL = 3.0                # во столько раз быстрее воздух набирается у поверхности
DIVE_V = 0.6                # м/с - вниз и вверх
SEE_FISH = 0.3              # м по глубине - видит рыбу близко
BREATHE = 2.0               # с у поверхности между нырками
PASS = 4.0                  # с - в среднем столько ждать, пока рыба проплывёт рядом


class MaskHeads:
    def __init__(self, seed=0):
        # вход: глубина пруда, глубина рыбы (доли 3 м), воздух (доля), усталость
        self.skill = GaussianSkill(4, [0.1, 1.0], [1.0, 18.0], hidden=12, seed=seed, sigma=0.25, lr=5e-3)
        self.dives = 0
        self.seen = 0
        self.history = []

    def plan(self, pond_depth, fish_depth, air, tired):
        feats = np.array([pond_depth / 3.0, fish_depth / 3.0, air / AIR, tired], np.float32)
        cmd, rec = self.skill.act(feats)
        depth = float(cmd[0]) * pond_depth           # доля глубины пруда
        return depth, float(cmd[1]), rec

    def finish(self, rec, seen, ran_out, learn=True):
        r = float(seen) - (0.5 if ran_out else 0.0)
        self.dives += 1
        self.seen += int(seen)
        self.history = (self.history + [float(seen)])[-50:]
        if learn:
            self.skill.learn(rec, r)
        return r

    def save(self, path):
        self.skill.save(path)

    def load(self, path):
        self.skill.load(path)
        return self


def dive_outcome(depth, hold, fish_depth, air, rng):
    """(увидел ли рыбу, воздух кончился ли, сколько секунд под водой). Рыба видна, если глубина
    нырка близко к её глубине и за время у дна она успела проплыть мимо."""
    under = 2.0 * depth / DIVE_V + hold
    ran_out = under > air
    held = hold if not ran_out else max(0.0, air - 2.0 * depth / DIVE_V)
    near = abs(depth - fish_depth) <= SEE_FISH
    passed = rng.random() < 1.0 - math.exp(-held / PASS)
    return bool(near and passed), ran_out, min(under, air)


def practice(heads, rng, learn=True):
    """Один нырок без жизни вокруг (законы, замеры): пруд 1.2-3 м, рыба на своей глубине, воздух
    после прошлых нырков - сколько осталось. Возвращает 1, если увидел рыбу."""
    pond = float(rng.uniform(1.2, 3.0))
    fish = float(rng.uniform(0.3, pond - 0.1))
    air = float(rng.uniform(0.5, 1.0)) * AIR
    depth, hold, rec = heads.plan(pond, fish, air, float(rng.uniform(0.0, 0.5)))
    seen, ran_out, _ = dive_outcome(depth, hold, fish, air, rng)
    heads.finish(rec, seen, ran_out, learn)
    return float(seen)
