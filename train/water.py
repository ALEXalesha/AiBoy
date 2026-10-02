"""Эволюция сети воды (world/water.py): веса отбираются, градиента нет.

    python -m train.water [--generations 40] [--population 24]

«Интересная» вода (спецификация 2.0, раздел 5), на мирах всех размеров (80-1280 м):
- около 60 % миров с водой (штраф за отклонение доли);
- в мире с водой от 1 до 4 водоёмов, в больших мирах (640-1280 м) до 6 и 8 (замер: со штрафом
  «больше 4 на 160 м» эволюция давала до 28 водоёмов в мире 1280 м);
- разные глубины: и мелкие, и глубокие (разброс глубин засчитывается);
- вода не заливает мир: больше 25 % ширины под водой - штраф.
Каждое поколение - новые зёрна. Итог - models/water.npz и models/water.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from game import models  # noqa: E402
from world import water  # noqa: E402
from world.generate import generate  # noqa: E402

WIDTHS = (80.0, 160.0, 320.0, 640.0, 1280.0)
WORLDS_PER_GEN = 30
ELITE = 6
TARGET_WET = 0.6


def max_ponds(width):
    return 4 if width <= 320 else (6 if width <= 640 else 8)


def stats(worlds, g):
    rows = []
    for w in worlds:
        ps = water.ponds(w, g)
        covered = sum(((p.right - p.left) % w.width) for p in ps) / w.width
        rows.append({"width": w.width, "ponds": len(ps), "depths": [p.depth for p in ps], "covered": covered,
                     "deep": sum(p.depth > water.WADE for p in ps)})
    return rows


def fitness(worlds, g):
    rows = stats(worlds, g)
    wet = [r for r in rows if r["ponds"]]
    share = len(wet) / len(rows)
    s = -3.0 * abs(share - TARGET_WET)
    for r in wet:
        s -= 1.0 * max(0.0, r["ponds"] - max_ponds(r["width"])) / len(rows)
        s -= 2.0 * max(0.0, r["covered"] - 0.25) / len(rows)
    depths = [d for r in wet for d in r["depths"]]
    if depths:
        s += 0.5 * min(float(np.std(depths)), 1.0) + 0.3 * min(float(np.mean([d > water.WADE for d in depths])), 0.5)
    return s


def check(worlds, g):
    rows = stats(worlds, g)
    wet = [r for r in rows if r["ponds"]]
    depths = [d for r in wet for d in r["depths"]]
    return {"worlds": len(rows), "wet_share": round(len(wet) / len(rows), 3),
            "ponds_per_wet_world": round(float(np.mean([r["ponds"] for r in wet])), 2) if wet else 0,
            "ponds_max": max([r["ponds"] for r in rows] or [0]),
            "worlds_over_limit": sum(r["ponds"] > max_ponds(r["width"]) for r in rows),
            "depth_mean": round(float(np.mean(depths)), 2) if depths else 0,
            "depth_max": round(float(np.max(depths)), 2) if depths else 0,
            "deep_share": round(float(np.mean([d > water.WADE for d in depths])), 2) if depths else 0,
            "covered_max": round(max([r["covered"] for r in rows] or [0]), 3),
            "deep_per_world_max": max([r["deep"] for r in rows] or [0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--population", type=int, default=24)
    args = ap.parse_args()
    wg = models.load().world_genome
    rng = np.random.default_rng(0)
    pop = [rng.normal(0.0, 1.0, water.SIZE) for _ in range(args.population)]
    sig = [0.3] * len(pop)
    started = time.perf_counter()
    test_worlds = [generate(wg, 7000 + i, WIDTHS[i % len(WIDTHS)]) for i in range(40)]
    before = check(test_worlds, pop[0])
    curve = []
    for gen in range(args.generations):
        seeds = rng.integers(1, 2 ** 31 - 1, WORLDS_PER_GEN)
        worlds = [generate(wg, int(s), WIDTHS[i % len(WIDTHS)]) for i, s in enumerate(seeds)]
        fit = [fitness(worlds, g) for g in pop]
        order = np.argsort(fit)[::-1]
        curve.append(round(float(fit[order[0]]), 4))
        print(f"поколение {gen + 1}: лучший {fit[order[0]]:.3f}, {time.perf_counter() - started:.0f} с", flush=True)
        elite, esig = [pop[i] for i in order[:ELITE]], [sig[i] for i in order[:ELITE]]
        pop, sig = list(elite), list(esig)
        while len(pop) < args.population:
            k = int(rng.integers(ELITE))
            s = esig[k] * float(np.exp(rng.normal(0, 0.3)))
            pop.append(elite[k] + rng.normal(0, s, water.SIZE))
            sig.append(s)
    best = pop[0]
    after = check(test_worlds, best)
    np.savez(model_file("water.npz"), genome=best)
    write_passport("water", {
        "what": "сеть воды: есть ли вода в мире и до какой доли глубины налита каждая котловина",
        "criterion": "около 60 % миров с водой, 1-4 водоёма на 160 м, разные глубины, под водой не больше 25 %",
        "generations": args.generations, "population": args.population,
        "wall_minutes": round((time.perf_counter() - started) / 60, 1), "curve": curve,
        "check_40_new_worlds": {"random_net": before, "evolved": after},
    })
    print("до:", before)
    print("после:", after)


if __name__ == "__main__":
    main()
