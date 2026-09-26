"""Отбор генератора человечков: из случайных геномов - самый разнообразный.

    python -m train.body [--candidates 300]

Веса генератора тела не учатся. Для каждого кандидата строятся 24 человечка разных зёрен;
разнообразие - средняя попарная разница пропорций (в долях допустимого размаха) и цветов
одежды, волос и кожи; плюс сколько причёсок из четырёх встретилось. Берётся лучший.
Итог - models/body.npz и models/body.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from body.cppn import LIMITS, generate_body, random_genome  # noqa: E402

SEEDS = 24


def features(b):
    sizes = [(getattr(b, k) - lo) / (hi - lo) for k, (lo, hi) in LIMITS.items()]
    return np.array(sizes + list(b.skin) + list(b.hair) + list(b.shirt) + list(b.pants))


def diversity(genome):
    bodies = [generate_body(genome, s) for s in range(SEEDS)]
    f = np.stack([features(b) for b in bodies])
    d = np.abs(f[:, None, :] - f[None, :, :]).mean(axis=2)
    pair = d[np.triu_indices(SEEDS, 1)].mean()
    styles = len({b.hair_style for b in bodies}) / 4.0
    names = len({b.name for b in bodies}) / SEEDS
    return float(pair + 0.1 * styles + 0.05 * names), bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    best, best_score, scores = None, -1.0, []
    for _ in range(args.candidates):
        g = random_genome(rng)
        score, _ = diversity(g)
        scores.append(score)
        if score > best_score:
            best, best_score = g, score
    _, bodies = diversity(best)
    np.savez(model_file("body.npz"), genome=best)
    write_passport("body", {
        "what": "генератор человечка по зерну: пропорции, толщины, цвета, причёска, имя",
        "how": "веса не учатся; из случайных геномов взят самый разнообразный",
        "criterion": "средняя попарная разница пропорций (доли допустимого размаха) и цветов по 24 зёрнам "
                     "+ 0.1 * доля встреченных причёсок + 0.05 * доля разных имён",
        "limits_m": {k: list(v) for k, v in LIMITS.items()},
        "candidates": args.candidates,
        "best_score": round(best_score, 4),
        "median_score": round(float(np.median(scores)), 4),
        "seconds": round(time.time() - t0, 1),
        "examples": [{"name": b.name, "height_m": round(b.height(), 3), "hair_style": b.hair_style}
                     for b in bodies[:10]],
    })
    print(f"лучший {best_score:.3f}, медиана {np.median(scores):.3f}")


if __name__ == "__main__":
    main()
