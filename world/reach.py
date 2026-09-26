"""Можно ли выбраться: куда человечек может попасть из точки мира - шагом или прыжком.

Способности тела измерены на физике (body/physics.py), а не придуманы:

- шагом он поднимается по склону не круче CLIMB (м на м), с запасом: заученная походка с
  высоко поднятой ногой уверенно проходит 0.5, на 0.9-1.1 ползёт еле-еле (стопа
  упирается в склон раньше, чем шаг сделан), на 1.5 застревает; походка мозга хуже;
- прыжком поднимается на JUMP_UP: прыжок подбрасывает таз на 0.91 м, с запасом - 0.8 м,
  и на JUMP_REACH по горизонтали;
- вниз можно всегда.

Мир - кольцо отсчётов через dx. Ловушка - яма, из которой нельзя попасть во все точки
мира ни шагом, ни прыжком.
"""
from collections import deque

import numpy as np

CLIMB = 0.5          # м подъёма на метр пути - шагом
JUMP_UP = 0.8        # м - прыжком вверх
JUMP_REACH = 1.0     # м - прыжком вперёд


def edges(h, dx):
    """Рёбра графа «отсюда можно попасть туда»: массивы (откуда, куда)."""
    h = np.asarray(h, float)
    n = len(h)
    idx = np.arange(n)
    src, dst = [], []
    for d in (-1, 1):
        j = (idx + d) % n
        ok = h[j] - h <= CLIMB * dx
        src.append(idx[ok])
        dst.append(j[ok])
    reach = max(1, int(round(JUMP_REACH / dx)))
    for sgn in (-1, 1):
        top = np.full(n, -np.inf)
        for d in range(1, reach + 1):
            j = (idx + sgn * d) % n
            top = np.maximum(top, h[j])
            if d >= 2:
                ok = top - h <= JUMP_UP
                src.append(idx[ok])
                dst.append(j[ok])
    return np.concatenate(src), np.concatenate(dst)


def _bfs(n, src, dst, start):
    adj = [[] for _ in range(n)]
    for a, b in zip(src.tolist(), dst.tolist()):
        adj[a].append(b)
    seen = np.zeros(n, bool)
    seen[start] = True
    q = deque([start])
    while q:
        i = q.popleft()
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                q.append(j)
    return seen


def reachable(h, dx, start):
    """Маска точек, куда можно попасть из start."""
    src, dst = edges(h, dx)
    return _bfs(len(h), src, dst, start)


def local_minima(h):
    h = np.asarray(h, float)
    return np.flatnonzero((h <= np.roll(h, 1)) & (h < np.roll(h, -1)))


def escapes(h, dx):
    """Маска точек, откуда можно попасть во весь мир.

    Если из самой высокой точки достижимо всё, то точка v может попасть всюду ровно
    тогда, когда может попасть в неё: это обратный обход графа из самой высокой точки."""
    h = np.asarray(h, float)
    n = len(h)
    src, dst = edges(h, dx)
    top = int(np.argmax(h))
    if not _bfs(n, src, dst, top).all():
        return np.array([_bfs(n, src, dst, i).all() for i in range(n)]) if n <= 64 else np.zeros(n, bool)
    return _bfs(n, dst, src, top)


def worst_hollow(world, look=6.0):
    """Самая глубокая впадина мира: (x дна, x левой кромки, x правой кромки, глубина).
    Кромки - самые высокие точки в look метрах по сторонам, глубина - по низкой кромке."""
    h = world.heights
    n, dx = len(h), world.dx
    k = int(round(look / dx))
    best = None
    for i in local_minima(h):
        left = [(i - d) % n for d in range(1, k + 1)]
        right = [(i + d) % n for d in range(1, k + 1)]
        li = max(left, key=lambda j: h[j])
        ri = max(right, key=lambda j: h[j])
        depth = min(h[li], h[ri]) - h[i]
        if best is None or depth > best[3]:
            lx = i * dx - ((i - li) % n) * dx
            rx = i * dx + ((ri - i) % n) * dx
            best = (i * dx, lx, rx, float(depth))
    return best


def traps(world):
    """Дно каждой ямы, откуда не попасть во весь мир: список индексов."""
    ok = escapes(world.heights, world.dx)
    return [int(i) for i in local_minima(world.heights) if not ok[i]]


def open_fraction(world):
    """Доля мира, откуда можно попасть везде: 1 - ловушек нет (мерило для эволюции)."""
    return float(escapes(world.heights, world.dx).mean())
