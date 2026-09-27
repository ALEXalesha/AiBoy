"""Стартовый навигатор: он гуляет по череде миров, и навигатор учится на ходу - так же,
как в игре. Мозг ходьбы при этом стартовый и не учится (быстрее; навигатору важно, куда
идти, а не как).

    python -m train.navigator [--worlds 24] [--minutes 6]

Миры разной ширины (80, 160, 320 м). Каждые eval-every миров - замер на отдельных мирах
(game/probe.wander, без обучения): доля «вперёд» на первом круге, развороты по своей воле
после, «туда-сюда». Обучение на ходу скачет (замер: после 8 и 16 миров - 1 разворот в
минуту в изученном мире, после 24 - 4), поэтому в models/navigator.npz берётся лучший
снимок по счёту score(); числа - в models/navigator.json и в плане.
"""
import argparse
import copy
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from acts.explore.navigator import Navigator  # noqa: E402
from game import models, probe  # noqa: E402

WIDTHS = (80, 160, 320)
EVAL = [(900 + i, WIDTHS[i % 3]) for i in range(6)]


def evaluate(m, nav, minutes=5):
    rows = []
    for ws, width in EVAL:
        frozen = copy.deepcopy(nav)
        rows.append(probe.wander(m, models.starter_brain(), ws, width, minutes, navigator=frozen, learn=False,
                                 nav_learn=False))
    keys = ("first_lap_forward", "first_lap_turns", "voluntary_after", "per_min_after", "back_forth", "looks_after")
    res = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}
    # доходит ли до места, куда тянет (метка «гитара» в случайном месте)
    fetched = []
    for ws, width in EVAL[:3]:
        frozen = copy.deepcopy(nav)
        fetched.append(probe.wander(m, models.starter_brain(), ws + 50, width, minutes, navigator=frozen,
                                    learn=False, nav_learn=False, pull_mark=True)["fetched"])
    res["fetched"] = round(float(np.mean(fetched)), 3)
    return res, rows


def score(res, rows):
    """Больше - лучше: в новом мире вперёд, в изученном - развороты есть в каждом мире,
    но около одного в минуту, без «туда-сюда»."""
    s = 2.0 * res["first_lap_forward"] - 0.5 * res["first_lap_turns"]
    s -= 1.0 * sum(1 for r in rows if r["voluntary_after"] == 0)
    s -= abs(res["per_min_after"] - 1.0) + 2.0 * max(0.0, res["per_min_after"] - 1.5)
    s -= 2.0 * res["back_forth"]
    s += 0.5 * res["fetched"]               # доходит до места, куда тянет (гитара)
    return round(s, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=24)
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--eval-every", type=int, default=8)
    args = ap.parse_args()
    m = models.load()
    nav = Navigator(seed=0)
    started = time.perf_counter()
    history = []
    before, rows = evaluate(m, nav)
    best = (score(before, rows), 0, copy.deepcopy(nav))
    print("до обучения:", before, "счёт", best[0], flush=True)
    history.append({"worlds": 0, "score": best[0], **before})
    for i in range(args.worlds):
        ws, width = 100 + i, WIDTHS[i % 3]
        probe.wander(m, models.starter_brain(), ws, width, args.minutes, navigator=nav, learn=False, nav_learn=True,
                     pull_mark=i % 2 == 1)
        print(f"мир {i + 1}: {ws}, {width} м, обновлений {nav.updates}, база {nav.baseline:.2f}, "
              f"{time.perf_counter() - started:.0f} с", flush=True)
        if (i + 1) % args.eval_every == 0 or i + 1 == args.worlds:
            res, rows = evaluate(m, nav)
            sc = score(res, rows)
            history.append({"worlds": i + 1, "score": sc, **res})
            print("замер:", res, "счёт", sc, flush=True)
            if sc > best[0]:
                best = (sc, i + 1, copy.deepcopy(nav))
    best[2].save(model_file("navigator.npz"))
    print("взят снимок после", best[1], "миров, счёт", best[0])
    write_passport("navigator", {
        "what": "навигатор - голова «гулять»: вперёд / назад / осмотреться по памяти мест",
        "training": {"worlds": args.worlds, "minutes_per_world": args.minutes, "widths": list(WIDTHS),
                     "wall_minutes": round((time.perf_counter() - started) / 60, 1), "updates": best[2].updates},
        "eval_worlds": EVAL, "history": history, "chosen_after_worlds": best[1], "chosen_score": best[0],
    })


if __name__ == "__main__":
    main()
