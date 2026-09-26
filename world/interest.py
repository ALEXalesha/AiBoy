"""Интересность мира - по ней эволюция отбирает веса сети мира (train/world.py).

Части, каждая от 0 до 1:

- relief - разнообразие высот: стандартное отклонение от 0.8 до 3 м - лучше всего;
- passable - проходимость: доля шагов по 0.5 м с перепадом не больше 0.45 м (ступенька,
  которую нога берёт без прыжка): от 50 % - ноль до 95 % - полный балл; и каждая стена
  (перепад больше 1 м на 0.5 м - не перепрыгнуть) на 100 м отнимает шестую часть;
- contrast - небо у горизонта и земля различаются по яркости;
- entities - сущностей от 6 до 20 на 100 м;
- kinds - хотя бы 4 разных вида;
- pop - сущности видно на фоне земли.
"""
import numpy as np

STEP = 0.45            # м - ступенька без прыжка
WALL = 1.0             # м на 0.5 м - уже не перепрыгнуть
WEIGHTS = {"relief": 1.0, "passable": 2.0, "contrast": 1.0, "entities": 1.0, "kinds": 1.0, "pop": 1.0}


def luma(rgb):
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def trapezoid(v, a, b, c, d):
    """0 до a, рост до b, 1 до c, спад до d."""
    if v <= a or v >= d:
        return 0.0
    if v < b:
        return (v - a) / (b - a)
    if v <= c:
        return 1.0
    return (d - v) / (d - c)


def _steps(world):
    k = max(1, int(round(0.5 / world.dx)))
    h = world.heights
    return np.abs(np.roll(h, -k) - h)


def passable_fraction(world):
    return float((_steps(world) <= STEP).mean())


def walls_per_100m(world):
    """Сколько раз по кругу мира перепад переходит за стену (подряд идущие - одна стена)."""
    high = _steps(world) > WALL
    starts = int(np.count_nonzero(high & ~np.roll(high, 1)))
    if high.all():
        starts = max(starts, 1)
    return starts * 100.0 / world.width


def score(world):
    h = world.heights
    parts = {"relief": trapezoid(float(h.std()), 0.15, 0.8, 3.0, 5.0)}
    walk = float(np.clip((passable_fraction(world) - 0.5) / 0.45, 0, 1))
    parts["passable"] = walk * float(np.clip(1.0 - walls_per_100m(world) / 6.0, 0, 1))
    p = world.palette
    parts["contrast"] = float(np.clip(abs(luma(p["sky_bottom"]) - luma(p["ground"])) / 0.3, 0, 1))
    per100 = len(world.entities) * 100.0 / world.width
    parts["entities"] = trapezoid(per100, 2.0, 6.0, 20.0, 32.0)
    parts["kinds"] = min(1.0, len({e.kind for e in world.entities}) / 4.0)
    if world.entities:
        g = np.array(p["ground"])
        dist = [float(np.abs(np.array(e.color) - g).sum()) for e in world.entities]
        parts["pop"] = float(np.clip(np.mean(dist) / 0.6, 0, 1))
    else:
        parts["pop"] = 0.0
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS.values())
    parts["total"] = float(np.clip(total, 0.0, 1.0))
    return parts


def diversity(worlds):
    """Насколько миры разных зёрен не похожи друг на друга: средняя разница цветов
    палитры и формы рельефа по парам, от 0 до 1."""
    if len(worlds) < 2:
        return 0.0
    vals = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            a, b = worlds[i], worlds[j]
            col = np.mean([np.abs(np.array(a.palette[k]) - np.array(b.palette[k])).sum() / 3
                           for k in ("sky_top", "sky_bottom", "ground", "grass")])
            n = min(len(a.heights), len(b.heights))
            rel = np.abs(a.heights[:n] - b.heights[:n]).mean() / 2.0
            vals.append(min(1.0, 0.5 * min(1.0, col / 0.3) + 0.5 * min(1.0, rel)))
    return float(np.mean(vals))
