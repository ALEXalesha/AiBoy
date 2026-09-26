"""Замер: как быстро музыкант учит мелодию. A (похожесть на мелодию) по попыткам.

    python tools\\measure_learning.py [--tune "Чижик-пыжик"] [--attempts 1500] [--seed 0]

Без владельца: награда = A и B (оценок C нет). Печатает медиану A по окнам в 50 попыток
и время. Числа идут в план и README.
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

from music import musician as mus  # noqa: E402
from music import reward, songbook  # noqa: E402
from music.reward import Played  # noqa: E402


def played_of(take):
    from music import guitar
    return [Played(t, guitar.midi(s, f), d, force) for t, s, f, force, d in mus.notes_of(take.actions, take.steps)]


def run(tune="Чижик-пыжик", attempts=1500, seed=0, phrase=0, lr=None, entropy=None, log=print):
    if lr is not None:
        mus.LR = lr
    if entropy is not None:
        mus.ENTROPY = entropy
    ph = songbook.phrases(songbook.by_name(tune))[phrase]
    m = mus.Musician(seed=seed)
    history, closeness = [], []
    started = time.perf_counter()
    for i in range(attempts):
        take = m.play(ph.steps, ph.meter, ph.notes)
        played = played_of(take)
        a = reward.closeness(played, ph.notes)
        b = reward.pleasant(played, ph.meter, history, ph.steps).total
        history = (history + [played])[-reward.HISTORY:]
        m.learn(take, reward.total(a, b, None), key=f"{tune}/{phrase}")
        closeness.append(a)
        if (i + 1) % 100 == 0:
            log(f"{i + 1}: медиана A за последние 100 - {np.median(closeness[-100:]):.3f}, "
                f"{time.perf_counter() - started:.0f} с")
    return closeness, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="Чижик-пыжик")
    ap.add_argument("--attempts", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--entropy", type=float)
    args = ap.parse_args()
    run(args.tune, args.attempts, args.seed, lr=args.lr, entropy=args.entropy)


if __name__ == "__main__":
    main()
