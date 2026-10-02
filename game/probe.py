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


def obstacle_ahead(world, x, facing):
    """Стена или обрыв в 2 м впереди - разворот там вынужденный, не по своей воле."""
    g0 = world.height_at(x)
    ahead = [world.height_at(x + facing * d) - g0 for d in np.arange(0.25, 2.01, 0.25)]
    return max(ahead) > 0.8 or min(ahead) < -1.5


def wander(models, brain, world_seed, width, minutes, navigator=None, body_seed=7, learn=True, nav_learn=None,
           explored_after=None, pull_mark=False, chest=()):
    """Гуляет minutes минут, как в игре. Возвращает развороты: на первом круге и потом (по
    своей воле и у препятствия), «туда-сюда» (два разворота за 5 с), осмотры, круги и долю
    времени на первом круге, когда он шёл в начальную сторону. Сундук по умолчанию пуст -
    меряется ходьба (как в 1.1.0, где гитары в замере не было); у игр - свои законы."""
    life = Life(models, world_seed, body_seed, width, brain, seed=world_seed, navigator=navigator, chest=chest)
    life.learn = learn
    life.nav_learn = nav_learn
    h = life.human
    lap = life.world.width if explored_after is None else explored_after
    start_odo, f0 = h.odometer, h.facing
    turns = []                  # (время, по своей воле, уже изучен)
    first_lap_time = first_lap_fwd = 0.0
    looks_after = 0
    rng = np.random.default_rng(world_seed + 77)
    fetched, pull_clock = 0, 0.0
    if pull_mark:
        # метка в случайном месте; то тянет к ней, то нет - как желание поиграть на гитаре,
        # лежащей там (своё имя: настоящую гитару он подбирает, и её метка пропадает)
        life.walk.places.marks["цель"] = float(rng.uniform(0, life.world.width))
    for i in range(int(minutes * 3600)):
        if pull_mark:
            pull_clock += 1 / 60
            if pull_clock >= 40.0:
                pull_clock = 0.0
                life.walk.pulls["цель"] = 1.0 if rng.random() < 0.5 else 0.0
            if "цель" in life.walk.reached and life.walk.pulls.get("цель", 0) > 0:
                fetched += 1
                life.walk.places.marks["цель"] = float(rng.uniform(0, life.world.width))
                life.walk.pulls["цель"] = 0.0
                life.walk.reached.discard("цель")
                life.walk.gap.pop("цель", None)
        x, f = h.px, h.facing
        looks = life.walk.looks
        life.tick()
        explored = h.odometer - start_odo >= lap
        if not explored:
            first_lap_time += 1
            first_lap_fwd += 1 if h.facing == f0 else 0
        elif life.walk.looks > looks:
            looks_after += 1
        if h.facing != f:
            turns.append((life.time, not obstacle_ahead(life.world, x, f), explored))
    later = [t for t in turns if t[2]]
    later_min = max(1e-9, minutes - first_lap_time / 3600)
    return {
        "laps": round((h.odometer - start_odo) / life.world.width, 2),
        "first_lap_turns": sum(1 for t in turns if not t[2]),
        # доля первого круга в одну сторону (какую - неважно: мог сразу повернуть и идти)
        "first_lap_forward": round(max(first_lap_fwd, first_lap_time - first_lap_fwd) / max(first_lap_time, 1), 3),
        "turns_after": len(later), "voluntary_after": sum(1 for t in later if t[1]),
        "per_min_after": round(len(later) / later_min, 2),
        "back_forth": sum(1 for a, b in zip(turns, turns[1:]) if b[0] - a[0] < 5.0),
        "looks_after": looks_after, "explored_minutes": round(later_min, 2), "fetched": fetched,
    }
