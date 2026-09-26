"""Стартовый мозг: несколько минут того же онлайн-обучения, что и в игре, в разных мирах.

    python -m train.brain [--minutes 10] [--world-seconds 60]

Мозг в игре учится на ходу сам. Чтобы человечек при первом запуске не был совсем
деревянным, при сборке он заранее живёт в череде миров (новый мир и новый человечек
каждые world-seconds секунд жизни) - обучение то же самое: любопытство, актёр-критик.
Сколько это заняло (минуты, шаги, часы жизни, миры) и что изменилось на 10 новых мирах
(путь, сдвиг с места, падения, прыжки) - честно в models/brain.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from body.cppn import generate_body  # noqa: E402
from body.physics import Human  # noqa: E402
from brain.brain import GAMMA, HIDDEN, LR, SIGMA, Brain, run_steps  # noqa: E402
from game import models  # noqa: E402
from world.generate import generate  # noqa: E402

WIDTH = 160.0
STEPS_PER_SECOND = 30


def life_in(brain, m, world_seed, body_seed, seconds, learn, x=None):
    world = generate(m.world_genome, world_seed, WIDTH)
    body = generate_body(m.body_genome, body_seed)
    human = Human(body, world, x=WIDTH * 0.5 if x is None else x)
    x0 = human.px
    brain.reset_memory()
    log = run_steps(brain, human, world, int(seconds * STEPS_PER_SECOND), learn=learn)
    ev = log["events"]
    return {
        "odometer_m": human.odometer,
        "displacement_m": abs(human.px - x0),
        "falls": ev.count("fall"), "jumps": ev.count("jump"),
        "pred_err": float(np.mean(log["pred_err"])) if log["pred_err"] else 0.0,
        "reward": float(np.mean(log["reward"])) if log["reward"] else 0.0,
    }


def evaluate(brain, m, seconds=60, worlds=10):
    rows = [life_in(brain, m, 5_000 + i, 6_000 + i, seconds, learn=False) for i in range(worlds)]
    return {
        "worlds": worlds, "seconds_each": seconds,
        "odometer_m_mean": round(float(np.mean([r["odometer_m"] for r in rows])), 2),
        "displacement_m_mean": round(float(np.mean([r["displacement_m"] for r in rows])), 2),
        "moved_over_2m": int(sum(r["displacement_m"] > 2.0 for r in rows)),
        "falls_total": int(sum(r["falls"] for r in rows)),
        "jumps_total": int(sum(r["jumps"] for r in rows)),
        "pred_err_mean": round(float(np.mean([r["pred_err"] for r in rows])), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--world-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="brain", help="имя модели в models/ (для опытов - другое)")
    args = ap.parse_args()
    m = models.load()
    rng = np.random.default_rng(0)
    before = evaluate(Brain(seed=0), m)
    print("до обучения:", before)
    brain = Brain(seed=0)
    t0 = time.time()
    history = []
    worlds = 0
    while time.time() - t0 < args.minutes * 60:
        row = life_in(brain, m, 90_000 + worlds, 91_000 + worlds, args.world_seconds, learn=True,
                      x=float(rng.uniform(0, WIDTH)))
        row["world"] = worlds + 1
        history.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        worlds += 1
        if worlds % 5 == 0:
            recent = history[-5:]
            print(f"миров {worlds}, {(time.time() - t0) / 60:.1f} мин: путь "
                  f"{np.mean([r['odometer_m'] for r in recent]):.1f} м, сдвиг "
                  f"{np.mean([r['displacement_m'] for r in recent]):.1f} м, награда "
                  f"{np.mean([r['reward'] for r in recent]):.3f}")
    seconds = time.time() - t0
    after = evaluate(brain, m)
    print("после:", after)
    brain.save(model_file(args.out + ".npz"))
    write_passport(args.out, {
        "what": "стартовые веса мозга: рекуррентный слой 46 -> 64, головы суставов, прыжка, "
                "критика и разговора, предсказатель 58 -> 96 -> 46",
        "how": "то же онлайн-обучение, что в игре (любопытство, актёр-критик), в череде миров",
        "honest_training_time": {
            "wall_minutes": round(seconds / 60, 1),
            "brain_steps": int(brain.steps),
            "lived_hours": round(brain.steps / STEPS_PER_SECOND / 3600, 2),
            "worlds": worlds,
            "world_seconds_each": args.world_seconds,
        },
        "hyper": {"hidden": HIDDEN, "gamma": GAMMA, "sigma": SIGMA, "lr": LR},
        "check_10_new_worlds_60s": {"before": before, "after": after},
        "history": history[::max(1, len(history) // 60)],
    })
    print(f"готово за {seconds / 60:.1f} мин, миров {worlds}, шагов {brain.steps}")


if __name__ == "__main__":
    main()
