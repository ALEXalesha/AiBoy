"""Эволюция макро-сети больших миров (world/macro.py): веса отбираются, градиента нет.

    python -m train.macro [--generations 30] [--population 20]

Приспособленность мира 640 или 1280 м (сеть мира - уже обученная, models/world.npz):
- без видимого повтора: минус больший из двух повторов - всего рельефа (автокорреляция на
  сдвигах 80-640 м) и формы холмов (world/macro.shape_repeat_score по холмам без
  макро-высоты: размах, меняясь, не прячет одинаковую форму);
- проходимо без защиты: минус 0.15 за каждое понижение рельефа на 15 % (генератор так
  убирает ямы-ловушки; если понижать приходится много, макро-рельеф слишком крут);
- есть на что посмотреть: перепад высот 8-20 м (хребты, котловины под озёра); меньше 8 м -
  штраф (замер: без него эволюция делала большие миры почти плоскими - перепад 4 м).
Каждое поколение - новые зёрна. Итог - models/macro.npz и models/macro.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from game import models  # noqa: E402
from world import macro, reach  # noqa: E402
from world.generate import generate  # noqa: E402

WIDTHS = (640.0, 1280.0)
SEEDS_PER_GEN = 6
ELITE = 5


def world_score(w):
    rep = max(macro.repeat_score(w.heights, w.dx), macro.shape_repeat_score(w.hills, w.dx))
    rng = float(np.ptp(w.heights))
    return -2.0 * rep - 0.15 * w.flattened + 1.5 * min(rng, 20.0) / 20.0 - 0.2 * max(0.0, 8.0 - rng)


def fitness(world_genome, g, seeds):
    """Среднее по мирам и худший мир: закон - для каждого мира, не в среднем."""
    sc = [world_score(generate(world_genome, s, WIDTHS[i % 2], macro_genome=g)) for i, s in enumerate(seeds)]
    return float(0.5 * np.mean(sc) + 0.5 * np.min(sc))


def check(world_genome, g, seeds):
    rows = []
    for i, s in enumerate(seeds):
        w = generate(world_genome, s, WIDTHS[i % 2], macro_genome=g)
        rows.append({"width": w.width, "repeat": macro.repeat_score(w.heights, w.dx),
                     "shape_repeat": macro.shape_repeat_score(w.hills, w.dx), "flattened": w.flattened,
                     "range_m": float(np.ptp(w.heights)), "traps": len(reach.traps(w))})
    out = {}
    for k in ("repeat", "shape_repeat", "flattened", "range_m"):
        v = [r[k] for r in rows]
        out[k] = {"mean": round(float(np.mean(v)), 3), "max": round(float(np.max(v)), 3),
                  "min": round(float(np.min(v)), 3)}
    out["worlds_with_traps"] = sum(r["traps"] > 0 for r in rows)
    out["worlds"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--population", type=int, default=20)
    args = ap.parse_args()
    wg = models.load().world_genome
    rng = np.random.default_rng(0)
    pop = [rng.normal(0.0, 1.0, macro.SIZE) for _ in range(args.population)]
    sigmas = [0.3] * len(pop)
    started = time.perf_counter()
    curve = []
    before = check(wg, pop[0], list(range(9000, 9020)))
    for gen in range(args.generations):
        seeds = [int(x) for x in rng.integers(1, 2 ** 31 - 1, SEEDS_PER_GEN)]
        fit = [fitness(wg, g, seeds) for g in pop]
        order = np.argsort(fit)[::-1]
        curve.append(round(float(fit[order[0]]), 4))
        print(f"поколение {gen + 1}: лучший {fit[order[0]]:.3f}, средний {np.mean(fit):.3f}, "
              f"{time.perf_counter() - started:.0f} с", flush=True)
        elite = [pop[i] for i in order[:ELITE]]
        esig = [sigmas[i] for i in order[:ELITE]]
        pop, sigmas = list(elite), list(esig)
        while len(pop) < args.population:
            k = int(rng.integers(ELITE))
            s = esig[k] * float(np.exp(rng.normal(0, 0.3)))
            pop.append(elite[k] + rng.normal(0, s, macro.SIZE))
            sigmas.append(s)
    best = pop[0]
    after = check(wg, best, list(range(9000, 9020)))
    np.savez(model_file("macro.npz"), genome=best)
    write_passport("macro", {
        "what": "макро-сеть миров 640 и 1280 м: изгиб координаты, хребты и котловины, размах",
        "criterion": "-2 x повтор (рельефа и формы холмов, сдвиги 80-640 м) - 0.15 x понижений + перепад 8-20 м",
        "generations": args.generations, "population": args.population,
        "wall_minutes": round((time.perf_counter() - started) / 60, 1), "curve": curve,
        "check_20_new_seeds": {"random_net": before, "evolved": after},
    })
    print("до:", before)
    print("после:", after)


if __name__ == "__main__":
    main()
