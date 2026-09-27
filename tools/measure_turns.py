"""Замер: разворачивается ли человечек сам и почему нет. Как в игре (обучение на ходу, шум
игры), без окна, 10 миров разных размеров по MINUTES минут игрового времени.

    python tools\\measure_turns.py [--minutes 12] [--navigator]

Печатает по миру: ширину, круги, развороты (сколько у препятствия - стена или обрыв
впереди, сколько по своей воле), развороты в минуту, «туда-сюда» (два разворота за 5 с),
любопытство на первом круге и потом.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np  # noqa: E402

from game import models as models_mod  # noqa: E402
from game.life import Life  # noqa: E402

WORLDS = [(11, 80), (12, 80), (13, 160), (14, 160), (15, 160), (16, 320), (17, 320), (18, 320), (19, 160), (20, 80)]
BODY = 7


def obstacle_ahead(world, x, facing):
    """Стена или обрыв в 2 м впереди - разворот там вынужденный."""
    g0 = world.height_at(x)
    ahead = [world.height_at(x + facing * d) - g0 for d in np.arange(0.25, 2.01, 0.25)]
    return max(ahead) > 0.8 or min(ahead) < -1.5


def run(world_seed, width, minutes, log=print):
    m = models_mod.load()
    life = Life(m, world_seed, BODY, width, models_mod.starter_brain(), seed=world_seed)
    h = life.human
    facing = h.facing
    turns = []                 # (время, по своей воле?)
    lap_len = life.world.width
    cur = {}                   # номер круга -> [награды любопытства]
    start = h.odometer
    for _ in range(int(minutes * 60 * 60)):
        before_x, before_f = h.px, h.facing
        events = life.tick()
        if h.facing != before_f:
            turns.append((life.time, not obstacle_ahead(life.world, before_x, before_f)))
        if life.decision is not None and life.brain.last_seen_err:
            lap = int((h.odometer - start) / lap_len)
            cur.setdefault(lap, []).append(life.brain.curiosity)
        del events
    vol = [t for t, v in turns if v]
    back_forth = sum(1 for a, b in zip(turns, turns[1:]) if b[0] - a[0] < 5.0)
    laps = (h.odometer - start) / lap_len
    c1 = float(np.mean(cur.get(0, [0]))) if cur else 0.0
    later = [x for k, v in cur.items() if k >= 1 for x in v]
    return dict(world=world_seed, width=width, laps=round(laps, 2), turns=len(turns), voluntary=len(vol),
                per_min=round(len(turns) / minutes, 2), back_forth=back_forth,
                curiosity_lap1=round(c1, 3), curiosity_later=round(float(np.mean(later)), 3) if later else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12)
    args = ap.parse_args()
    t0 = time.perf_counter()
    for ws, width in WORLDS:
        print(run(ws, width, args.minutes), f"{time.perf_counter() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
