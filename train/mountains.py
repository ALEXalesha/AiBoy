"""Эволюция сети гор (world/mountains.py): веса отбираются, градиента нет.

    python -m train.mountains [--generations 30] [--population 16]

Приспособленность (сеть мира и макро-сеть - уже обученные):
- в каждом мире 1280 м есть гора, примерно в половине - две; в мирах 640 м гора - примерно в
  половине (штраф за отклонение доли от цели);
- высоты разные: разброс высот по мирам (60-100 м) ценится;
- стены в обе стороны в разных мирах;
- проходимо: ни одной ловушки, и холмы понижать приходится редко (минус за каждое понижение);
- стена - настоящая: высота стены (уклон круче шага) не меньше 0.8 высоты горы.
Каждое поколение - новые зёрна. Итог - models/mountains.npz и models/mountains.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from game import models  # noqa: E402
from world import mountains, reach  # noqa: E402
from world.generate import generate  # noqa: E402

SEEDS_PER_GEN = 16
ELITE = 4


def rows_for(world_genome, g, seeds):
    rows = []
    old = mountains._default
    mountains._default = g                   # generate() берёт сеть гор по умолчанию
    try:
        for i, s in enumerate(seeds):
            width = 1280.0 if i % 2 == 0 else 640.0
            w = generate(world_genome, s, width)
            walls = [mountains.face_wall(w, p)[0] / p.height for p in w.mountains]
            rows.append({"width": width, "count": len(w.mountains), "planned": len(mountains.plan(width, s, g)),
                         "heights": [p.height for p in w.mountains], "steep": [p.steep for p in w.mountains],
                         "lowered": w.mountain_lowered, "traps": len(reach.traps(w)),
                         "wall": min(walls) if walls else 1.0})
    finally:
        mountains._default = old
    return rows


def score(rows):
    big = [r for r in rows if r["width"] == 1280.0]
    small = [r for r in rows if r["width"] == 640.0]
    sc = 0.0
    sc -= 2.0 * sum(r["count"] == 0 for r in big) / max(1, len(big))
    sc -= abs(np.mean([r["count"] == 2 for r in big]) - 0.5)
    sc -= abs(np.mean([r["count"] == 1 for r in small]) - 0.5)
    hs = [h for r in rows for h in r["heights"]]
    sc += 0.5 * (np.std(hs) / 12.0 if len(hs) > 1 else 0.0)
    st = [s for r in rows for s in r["steep"]]
    sc -= 1.5 * abs(np.mean(st)) if st else 1.5
    sc -= 0.15 * np.mean([r["lowered"] for r in rows])
    sc -= 3.0 * sum(r["traps"] > 0 or r["count"] < r["planned"] for r in rows)
    sc -= 2.0 * sum(r["wall"] < 0.8 for r in rows)
    return float(sc)


def check(world_genome, g, seeds):
    rows = rows_for(world_genome, g, seeds)
    big = [r for r in rows if r["width"] == 1280.0]
    small = [r for r in rows if r["width"] == 640.0]
    hs = [h for r in rows for h in r["heights"]]
    return {"worlds": len(rows), "1280_with_mountain": round(float(np.mean([r["count"] > 0 for r in big])), 2),
            "1280_with_two": round(float(np.mean([r["count"] == 2 for r in big])), 2),
            "640_with_mountain": round(float(np.mean([r["count"] > 0 for r in small])), 2),
            "height_min": round(float(min(hs)), 1) if hs else None, "height_max": round(float(max(hs)), 1) if hs else None,
            "steep_right_share": round(float(np.mean([s > 0 for r in rows for s in r["steep"]])), 2),
            "lowered_mean": round(float(np.mean([r["lowered"] for r in rows])), 2),
            "worlds_with_traps": sum(r["traps"] > 0 for r in rows),
            "dropped": sum(r["count"] < r["planned"] for r in rows),
            "weakest_wall_share": round(float(min(r["wall"] for r in rows)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=16)
    args = ap.parse_args()
    wg = models.load().world_genome
    rng = np.random.default_rng(0)
    pop = [rng.normal(0.0, 1.0, mountains.SIZE) for _ in range(args.population)]
    sigmas = [0.3] * len(pop)
    started = time.perf_counter()
    curve = []
    test_seeds = list(range(9300, 9340))
    before = check(wg, pop[0], test_seeds)
    for gen in range(args.generations):
        seeds = [int(x) for x in rng.integers(1, 2 ** 31 - 1, SEEDS_PER_GEN)]
        fit = [score(rows_for(wg, g, seeds)) for g in pop]
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
            pop.append(elite[k] + rng.normal(0, s, mountains.SIZE))
            sigmas.append(s)
    best = pop[0]
    after = check(wg, best, test_seeds)
    np.savez(model_file("mountains.npz"), genome=best)
    write_passport("mountains", {
        "what": "сеть гор миров 640 и 1280 м: сколько гор, высоты 60-100 м, в какую сторону стена, где",
        "criterion": "гора в каждом мире 1280 (две - в половине), в половине миров 640; разные высоты; стены в обе "
                     "стороны; ни ловушек, ни выброшенных гор; стена - не меньше 0.8 высоты горы",
        "generations": args.generations, "population": args.population,
        "wall_minutes": round((time.perf_counter() - started) / 60, 1), "curve": curve,
        "check_40_new_seeds": {"random_net": before, "evolved": after},
    })
    print("до:", before)
    print("после:", after)


if __name__ == "__main__":
    main()
