"""Эволюционная стратегия для стартовых весов мозга (часть train/brain.py).

Почему не только онлайн-обучение: замер 26.09 - 12 минут того же актёра-критика, что в
игре, давали человечка, который топчется на месте (сдвиг 1-6 м за минуту), а прежний
«быстрый» мозг ходил за счёт дыры в физике разворота. Градиент по одному шагу и одному
образцу слишком шумный, чтобы за минуты выучить согласованную походку.

Здесь та же цель - любопытство, но меренное грубо, по эпизоду: сколько нового мира он
увидел (размах пройденного по x), выбрался ли из самой глубокой впадины, - минус
развороты, дёрганье, падения и прыжки (он должен быть спокойным и ходить, а не скакать). Веса актёра (рекуррентный слой
и головы суставов и прыжка) меняются шумом, пары ±шум сравниваются, шаг - по разнице
(OpenAI-ES, антитетические пары, ранги). Код не знает, как ходить: он только сравнивает,
кто из двух почти одинаковых мозгов увидел больше.
"""
import math
import os

import numpy as np

EPISODE = 12.0            # с жизни на эпизод
PAIRS = 16                # пар ±шум на поколение
SIGMA_ES = 0.03
LR_ES = 0.03
JUMP_COST = 1.0           # м «увиденного» за прыжок: без этой цены он скакал 28 раз в минуту,
                          # с ценой 0.6 - 25 раз, с ценой 2.0 не прыгал совсем и не выбирался из впадин
JITTER_COST = 60.0        # за долю кадров, где сустав дёрнулся назад
HOLLOW_BONUS = 8.0        # м - выбрался из самой глубокой впадины

_models = None


def _init():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    global _models
    from game import models
    _models = models.load()


def layers(brain):
    return [brain.cell, brain.pi, brain.jump]


def get_flat(brain):
    return np.concatenate([p.ravel() for layer in layers(brain) for _, p, _ in layer.params()]).astype(np.float64)


def set_flat(brain, vec):
    k = 0
    for layer in layers(brain):
        for _, p, _ in layer.params():
            n = p.size
            p[...] = vec[k:k + n].reshape(p.shape)
            k += n


def episode(theta, world_seed, body_seed, hollow, seed):
    """Эпизод в условиях игры (шум PLAY_NOISE, без обучения). Счёт и замеры."""
    from body import skeleton
    from body.cppn import generate_body
    from body.physics import Human
    from brain.brain import PHYSICS_PER_STEP, PLAY_NOISE, Brain
    from brain.observe import observe
    from world import reach
    from world.generate import generate

    m = _models
    world = generate(m.world_genome, world_seed, 160.0)
    body = generate_body(m.body_genome, body_seed)
    brain = Brain(seed=seed)
    set_flat(brain, theta)
    if hollow:
        bottom, left, right, _ = reach.worst_hollow(world)
        x0 = bottom
    else:
        x0 = 80.0
        left = right = None
    human = Human(body, world, x=x0)
    rel = skeleton.feet(body, human.angles, human.facing)
    human.py = human._req(human.px, rel)
    human.anchor = [human.px + fx for fx, _ in rel]
    lo = hi = human.px
    last_facing, turns, falls, jumps = human.facing, 0, 0, 0
    prev_a, prev_d, flips, moves = human.angles.copy(), None, 0, 0
    escaped_at = None
    steps = int(EPISODE * 30)
    for i in range(steps):
        d, _ = brain.step(observe(human, world), learn=False, noise=PLAY_NOISE)
        for k in range(PHYSICS_PER_STEP):
            ev = human.step(d.targets, d.turn, d.jump and k == 0)
            falls += ev.count("fall")
            jumps += ev.count("jump")
            delta = human.angles - prev_a
            prev_a = human.angles.copy()
            if prev_d is not None:
                mv = (np.abs(delta) > 1e-4) & (np.abs(prev_d) > 1e-4)
                flips += int(np.count_nonzero((np.sign(delta) != np.sign(prev_d)) & mv))
                moves += delta.size
            prev_d = delta
        if human.facing != last_facing:
            turns += 1
            last_facing = human.facing
        lo, hi = min(lo, human.px), max(hi, human.px)
        if hollow and escaped_at is None and (human.px <= left or human.px >= right):
            escaped_at = i / 30.0
    jitter = flips / max(1, moves)
    seen = hi - lo
    score = seen - 0.5 * turns - JITTER_COST * jitter - 3.0 * falls - JUMP_COST * jumps
    if hollow:
        score += HOLLOW_BONUS if escaped_at is not None else 0.0
    return score, {"seen_m": seen, "turns": turns, "jitter": jitter, "falls": falls, "jumps": jumps,
                   "escaped": escaped_at is not None}


def _job(args):
    return episode(*args)


class ES:
    def __init__(self, theta, rng):
        self.theta = theta.copy()
        self.rng = rng
        self.m = np.zeros_like(theta)
        self.v = np.zeros_like(theta)
        self.t = 0

    def generation(self, pool, worlds):
        """worlds - [(мир, тело, во впадине?)]: каждый вариант живёт во всех этих мирах."""
        eps = self.rng.normal(size=(PAIRS, len(self.theta)))
        jobs, seed = [], int(self.rng.integers(1 << 30))
        for i in range(PAIRS):
            for sign in (1, -1):
                th = self.theta + sign * SIGMA_ES * eps[i]
                for ws, bs, hollow in worlds:
                    jobs.append((th, ws, bs, hollow, seed))
        res = pool.map(_job, jobs)
        per = len(worlds)
        scores = np.array([np.mean([r[0] for r in res[j * per:(j + 1) * per]]) for j in range(2 * PAIRS)])
        ranks = np.empty(len(scores))
        ranks[np.argsort(scores)] = np.arange(len(scores))
        ranks = ranks / (len(scores) - 1) - 0.5
        diff = ranks[0::2] - ranks[1::2]
        grad = (diff[:, None] * eps).sum(axis=0) / (PAIRS * SIGMA_ES)
        # Adam по направлению подъёма
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        mh = self.m / (1 - 0.9 ** self.t)
        vh = self.v / (1 - 0.999 ** self.t)
        self.theta += LR_ES * mh / (np.sqrt(vh) + 1e-8) - 0.001 * LR_ES * self.theta
        infos = [r[1] for r in res]
        return float(scores.mean()), {
            "seen_m": float(np.mean([x["seen_m"] for x in infos])),
            "turns": float(np.mean([x["turns"] for x in infos])),
            "jitter": float(np.mean([x["jitter"] for x in infos])),
            "jumps": float(np.mean([x["jumps"] for x in infos])),
            "escaped": float(np.mean([x["escaped"] for x, (_, _, h) in zip(infos, worlds * (2 * PAIRS)) if h]
                                     or [0.0])),
        }


def pool(n=4):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(n, initializer=_init)


__all__ = ["ES", "pool", "get_flat", "set_flat", "episode", "math"]
