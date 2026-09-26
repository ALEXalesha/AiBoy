"""Замеры поведения человечка в условиях игры (как в окне: game/life.py, шум PLAY_NOISE,
обучение на ходу) - для отбора стартового мозга, паспорта и тестов-законов.

- turns_per_min - сколько раз в минуту развернулся;
- jitter - доля кадров, где сустав поменял направление движения (дёрганье);
- dangle - средний |изменение угла| сустава за кадр, рад;
- odometer / displacement - путь со всеми шагами и сдвиг от места старта, м;
- escape - выбрался ли из самой глубокой впадины мира за отведённое время.
"""
import numpy as np

from body import skeleton
from game.life import Life
from world import reach

WIDTH = 160.0


def behaviour(models, brain_factory, world_seed, body_seed, seconds=60, learn=True, x=None):
    brain = brain_factory()
    life = Life(models, world_seed, body_seed, WIDTH, brain, seed=world_seed)
    life.learn = learn
    if x is not None:
        life.human.px = x
    x0 = life.human.px
    facing, angles = [life.human.facing], [life.human.angles.copy()]
    for _ in range(int(seconds * 60)):
        life.tick()
        facing.append(life.human.facing)
        angles.append(life.human.angles.copy())
    a = np.array(angles)
    d = np.diff(a, axis=0)
    moving = np.abs(d) > 1e-4
    flips = (np.sign(d[1:]) != np.sign(d[:-1])) & moving[1:] & moving[:-1]
    return {
        "turns_per_min": float(np.count_nonzero(np.diff(facing))) * 60.0 / seconds,
        "jitter": float(flips.mean()),
        "dangle": float(np.abs(d).mean()),
        "odometer_m": float(life.distance),
        "displacement_m": float(abs(life.human.px - x0)),
        "jumps": life.jumps, "falls": life.falls,
    }


def escape(models, brain_factory, world_seed, body_seed, seconds=30, learn=True):
    """Человечек на дне самой глубокой впадины: выбрался ли за seconds (таз за кромкой)."""
    brain = brain_factory()
    life = Life(models, world_seed, body_seed, WIDTH, brain, seed=world_seed)
    life.learn = learn
    bottom, left, right, depth = reach.worst_hollow(life.world)
    h = life.human
    h.px = bottom
    rel = skeleton.feet(h.body, h.angles, h.facing)
    h.py = h._req(h.px, rel)
    h.anchor = [h.px + fx for fx, _ in rel]
    for _ in range(int(seconds * 60)):
        life.tick()
        if h.px <= left or h.px >= right:
            return True, life.time, depth
    return False, life.time, depth


def summary(models, brain_factory, seeds, seconds=60, learn=True):
    rows = [behaviour(models, brain_factory, ws, bs, seconds, learn) for ws, bs in seeds]
    out = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
    out["worlds"] = len(rows)
    out["moved_over_2m"] = int(sum(r["displacement_m"] > 2.0 for r in rows))
    return out, rows
