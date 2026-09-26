"""Эволюция сети мира: веса CPPN отбираются на интересность мира, градиента нет.

    python -m train.world [--generations 120] [--population 40]

Каждое поколение - новые зёрна мира (одни на всё поколение), чтобы геном был хорош для
любого зерна, а не для шести знакомых. Приспособленность = средняя интересность мира
(world/interest.py) * 0.75 + разнообразие миров разных зёрен * 0.15 + доля ночных миров
около четверти * 0.1. Лучшие 8 переходят в следующее поколение как есть, остальные -
их мутанты (гауссов шум, у каждого своя сила), изредка - смесь двух родителей.

Итог - models/world.npz (геном) и models/world.json (паспорт: критерий, кривая, проверка
на 60 новых зёрнах).
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from world import cppn, interest, reach  # noqa: E402
from world.generate import generate  # noqa: E402

WIDTH = 160.0
SEEDS_PER_GEN = 6
ELITE = 8
NIGHT_TARGET = (0.15, 0.4)


def night_score(worlds):
    frac = np.mean([w.night for w in worlds])
    lo, hi = NIGHT_TARGET
    if lo <= frac <= hi:
        return 1.0
    return float(max(0.0, 1.0 - min(abs(frac - lo), abs(frac - hi)) / 0.3))


def fitness(genome, seeds):
    worlds = [generate(genome, s, WIDTH) for s in seeds]
    total = np.mean([interest.score(w)["total"] for w in worlds])
    return 0.75 * total + 0.15 * interest.diversity(worlds) + 0.1 * night_score(worlds)


def check(genome, seeds):
    worlds = [generate(genome, s, WIDTH) for s in seeds]
    scores = [interest.score(w) for w in worlds]
    return {
        "seeds": f"{seeds[0]}..{seeds[-1]}",
        "worlds": len(worlds),
        "interest_mean": round(float(np.mean([s["total"] for s in scores])), 4),
        "parts_mean": {k: round(float(np.mean([s[k] for s in scores])), 4)
                       for k in interest.WEIGHTS},
        "passable_fraction_mean": round(float(np.mean([interest.passable_fraction(w) for w in worlds])), 4),
        "passable_fraction_min": round(float(np.min([interest.passable_fraction(w) for w in worlds])), 4),
        "walls_per_100m_mean": round(float(np.mean([interest.walls_per_100m(w) for w in worlds])), 3),
        "worlds_with_traps": int(sum(bool(reach.traps(w)) for w in worlds)),
        "open_fraction_min": round(float(min(reach.open_fraction(w) for w in worlds)), 4),
        "height_std_mean": round(float(np.mean([w.heights.std() for w in worlds])), 3),
        "entities_per_100m_mean": round(float(np.mean([len(w.entities) * 100 / w.width for w in worlds])), 2),
        "night_fraction": round(float(np.mean([w.night for w in worlds])), 3),
        "diversity": round(interest.diversity(worlds[:20]), 4),
    }


def evolve(generations, population, seed=0, log=print, init=None):
    """init - геном, с которого начать (его мутанты - первое поколение); иначе случайные."""
    rng = np.random.default_rng(seed)
    if init is None:
        pop = [cppn.random_genome(rng, 0.6) for _ in range(population)]
    else:
        pop = [init.copy()] + [init + rng.normal(0, 0.05, len(init)) for _ in range(population - 1)]
    sigmas = rng.uniform(0.05, 0.3, population)
    curve = []
    best, best_fit = None, -1.0
    for gen in range(generations):
        seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, SEEDS_PER_GEN)]
        fits = np.array([fitness(g, seeds) for g in pop])
        order = np.argsort(-fits)
        curve.append({"generation": gen + 1, "best": round(float(fits[order[0]]), 4),
                      "mean": round(float(fits.mean()), 4)})
        if fits[order[0]] > best_fit * 0.98 or best is None:
            # лучший этого поколения - кандидат; окончательно решит проверка ниже
            best, best_fit = pop[order[0]].copy(), float(fits[order[0]])
        if gen % 10 == 0 or gen == generations - 1:
            log(f"поколение {gen + 1}: лучший {fits[order[0]]:.3f}, средний {fits.mean():.3f}")
        elites = [pop[i] for i in order[:ELITE]]
        el_sig = [sigmas[i] for i in order[:ELITE]]
        new, new_sig = [e.copy() for e in elites], list(el_sig)
        while len(new) < population:
            i = int(rng.integers(ELITE))
            child = elites[i].copy()
            if rng.random() < 0.2:
                other = elites[int(rng.integers(ELITE))]
                mask = rng.random(len(child)) < 0.5
                child[mask] = other[mask]
            s = float(np.clip(el_sig[i] * np.exp(rng.normal(0, 0.2)), 0.02, 0.5))
            child += rng.normal(0, s, len(child))
            new.append(child)
            new_sig.append(s)
        pop, sigmas = new, np.array(new_sig)
    # из элиты последнего поколения - тот, что лучше на 30 общих новых зёрнах
    final_seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, 30)]
    finals = [(fitness(g, final_seeds), g) for g in pop[:ELITE] + [best]]
    fit, genome = max(finals, key=lambda t: t[0])
    return genome, curve, float(fit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="геном .npz, с которого продолжить эволюцию")
    args = ap.parse_args()
    init = np.load(args.init)["genome"] if args.init else None
    started = time.time()
    rand = check(cppn.random_genome(np.random.default_rng(3), 1.0), list(range(1000, 1060)))
    genome, curve, fit = evolve(args.generations, args.population, args.seed, init=init)
    seconds = time.time() - started
    validation = check(genome, list(range(1000, 1060)))
    np.savez(model_file("world.npz"), genome=genome)
    write_passport("world", {
        "what": "веса CPPN мира (рельеф и палитра), отобранные эволюцией на интересность",
        "criterion": {
            "fitness": "0.75 * средняя интересность + 0.15 * разнообразие миров + 0.1 * доля ночи 15-40 %",
            "interest_parts": {
                "relief": "стандартное отклонение высот: 0.8-3 м - лучше всего",
                "passable": "доля шагов по 0.5 м с подъёмом <= 0.25 м (шагом; 50 % - ноль, 97 % - полный балл) "
                            "* (доля мира, откуда можно попасть везде шагом или прыжком)^2, при ловушке ещё * 0.3 "
                            "* (1 - стен на 100 м / 6)",
                "contrast": "разница яркости неба у горизонта и земли, 0.3 - полный балл",
                "entities": "сущностей 10-28 на 100 м",
                "kinds": "хотя бы 4 вида сущностей",
                "pop": "сущности отличаются цветом от земли",
            },
            "weights": {k: v for k, v in __import__("world.interest", fromlist=["WEIGHTS"]).WEIGHTS.items()},
        },
        "genome_size": int(len(genome)),
        "population": args.population,
        "generations": args.generations,
        "started_from": "геном прошлой эволюции (300 поколений, мерило со ступенькой 0.45 м) - "
                        "со случайного старта новое мерило не давало сигнала" if init is not None else "случайные геномы",
        "seeds_per_generation": SEEDS_PER_GEN,
        "elite": ELITE,
        "world_width_m": WIDTH,
        "final_fitness": round(fit, 4),
        "seconds": round(seconds, 1),
        "curve": curve,
        "validation_new_seeds": validation,
        "random_genome_same_check": rand,
    })
    print(f"готово за {seconds / 60:.1f} мин; проверка: {validation}")


if __name__ == "__main__":
    main()
