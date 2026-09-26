"""Стартовый мозг: несколько минут того же онлайн-обучения, что и в игре, в разных мирах.

    python -m train.brain [--minutes 12] [--world-seconds 60]

Мозг в игре учится на ходу сам. Чтобы человечек при первом запуске не был совсем
деревянным, при сборке он заранее живёт в череде миров (новый мир и новый человечек
каждые world-seconds секунд жизни) - обучение то же самое: любопытство, актёр-критик,
цена рывков; шум исследования - полный (в игре он в PLAY_NOISE раз меньше).

Поведение при обучении на ходу скачет, поэтому каждые SNAPSHOT_EVERY миров снимок весов
проверяется в условиях игры (game/probe.py: 5 отдельных миров по 30 с, шум игры, без
обучения) и в models/ берётся лучший по счёту: сдвиг с места минус штраф за развороты и
дёрганье. Сколько это заняло и что изменилось на 10 новых мирах - в models/brain.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from body.cppn import generate_body  # noqa: E402
from body.physics import Human  # noqa: E402
from brain.brain import GAMMA, HIDDEN, JERK, LR, PLAY_NOISE, SIGMA, Brain, run_steps  # noqa: E402
from game import models, probe  # noqa: E402
from train import es  # noqa: E402
from nn import io  # noqa: E402
from world.generate import generate  # noqa: E402

WIDTH = 160.0
STEPS_PER_SECOND = 30
SNAPSHOT_EVERY = 40
CHECK_SEEDS = [(3_000 + i, 3_100 + i) for i in range(5)]
EVAL_SEEDS = [(5_000 + i, 6_000 + i) for i in range(10)]


def train_life(brain, m, world_seed, body_seed, seconds, x):
    world = generate(m.world_genome, world_seed, WIDTH)
    human = Human(generate_body(m.body_genome, body_seed), world, x=x)
    x0 = human.px
    brain.reset_memory()
    log = run_steps(brain, human, world, int(seconds * STEPS_PER_SECOND), learn=True)
    return {"odometer_m": human.odometer, "displacement_m": abs(human.px - x0),
            "jumps": log["events"].count("jump"),
            "reward": float(np.mean(log["reward"])) if log["reward"] else 0.0}


def frozen(weights, steps):
    """Фабрика мозгов из снимка весов - каждый замер со своей копией, снимок не портится."""
    def make():
        b = Brain(seed=0)
        io.restore(b.nets(), weights)
        b.steps = steps
        return b
    return make


def score(s):
    return s["displacement_m"] - 0.3 * s["turns_per_min"] - 40.0 * s["jitter"] - 0.3 * s["jumps"]


def escapes(m, factory, seeds, seconds=30):
    rows = [probe.escape(m, factory, ws, bs, seconds) for ws, bs in seeds]
    return {"worlds": len(rows), "escaped": int(sum(r[0] for r in rows)),
            "mean_depth_m": round(float(np.mean([r[2] for r in rows])), 2),
            "mean_seconds_when_escaped": round(float(np.mean([r[1] for r in rows if r[0]] or [0])), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es-minutes", type=float, default=12.0)
    ap.add_argument("--minutes", type=float, default=3.0, help="онлайн-обучение предсказателя и критика")
    ap.add_argument("--world-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="brain", help="имя модели в models/ (для опытов - другое)")
    ap.add_argument("--init", help="мозг .npz, с которого продолжить эволюцию")
    ap.add_argument("--init-note", default="", help="откуда этот мозг - для паспорта")
    args = ap.parse_args()
    m = models.load()
    rng = np.random.default_rng(0)
    before, _ = probe.summary(m, lambda: Brain(seed=0), EVAL_SEEDS, 60, learn=True)
    print("до обучения:", before)

    # 1. эволюционная стратегия: веса актёра
    t0 = time.time()
    base = Brain(seed=0)
    if args.init:
        base.load(args.init)
    opt = es.ES(es.get_flat(base), rng)
    curve, snapshots = [], []
    best = None
    gen = 0
    with es.pool(4) as pool:
        while time.time() - t0 < args.es_minutes * 60:
            seeds = rng.integers(100_000, 900_000, 3)
            worlds = [(int(seeds[0]), int(seeds[0]) + 1, False), (int(seeds[1]), int(seeds[1]) + 1, True),
                      (int(seeds[2]), int(seeds[2]) + 1, True)]
            mean, info = opt.generation(pool, worlds)
            gen += 1
            curve.append({"generation": gen, "score": round(mean, 3), **{k: round(v, 3) for k, v in info.items()}})
            if gen % 5 == 0:
                print(f"поколение {gen}, {(time.time() - t0) / 60:.1f} мин: счёт {mean:.2f}, увидел "
                      f"{info['seen_m']:.1f} м, разворотов {info['turns']:.1f}, дёрганье {info['jitter']:.3f}, "
                      f"прыжков {info['jumps']:.1f}, "
                      f"выбрался {info['escaped']:.2f}")
            if gen % 25 == 0:
                b = Brain(seed=0)
                es.set_flat(b, opt.theta)
                weights = {k: v.copy() for k, v in io.collect(b.nets()).items()}
                sm, _ = probe.summary(m, frozen(weights, 0), CHECK_SEEDS, 30, learn=False)
                pits = escapes(m, frozen(weights, 0), CHECK_SEEDS)
                sm["score"] = round(score(sm) + 3.0 * pits["escaped"], 2)
                sm["generation"], sm["escaped"] = gen, pits["escaped"]
                snapshots.append(sm)
                print(f"  снимок поколения {gen}: сдвиг {sm['displacement_m']:.1f} м, разворотов "
                      f"{sm['turns_per_min']:.0f}/мин, дёрганье {sm['jitter']:.3f}, из впадин {pits['escaped']}/5")
                if best is None or sm["score"] > best[0]:
                    best = (sm["score"], opt.theta.copy(), gen)
    es_seconds = time.time() - t0
    if best is None:
        best = (0.0, opt.theta.copy(), gen)
    _, theta, chosen = best

    # 2. онлайн: учатся предсказатель и критик, актёр - как вывела эволюция
    brain = Brain(seed=0)
    es.set_flat(brain, theta)
    brain.learn_actor = False
    t1 = time.time()
    history = []
    worlds = 0
    while time.time() - t1 < args.minutes * 60:
        row = train_life(brain, m, 90_000 + worlds, 91_000 + worlds, args.world_seconds,
                         float(rng.uniform(0, WIDTH)))
        history.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        worlds += 1
    online_seconds = time.time() - t1
    brain.learn_actor = True
    brain.reset_memory()
    weights = {k: v.copy() for k, v in io.collect(brain.nets()).items()}
    factory = frozen(weights, brain.steps)
    after, _ = probe.summary(m, factory, EVAL_SEEDS, 60, learn=True)
    pits = escapes(m, factory, EVAL_SEEDS)
    print("после:", after, "; из впадин:", pits)
    brain.save(model_file(args.out + ".npz"))
    write_passport(args.out, {
        "what": "стартовые веса мозга: рекуррентный слой 48 -> 64, головы суставов, прыжка, "
                "критика и «сказать», предсказатель 60 -> 96 -> 48",
        "how": "1) эволюционная стратегия (train/es.py) по весам актёра: счёт эпизода - сколько нового "
               "мира он увидел (размах пройденного), выбрался ли из самой глубокой впадины, минус "
               "развороты, дёрганье, падения и прыжки; каждые 25 поколений снимок проверялся в условиях игры, "
               "взят лучший; 2) онлайн-обучение предсказателя и критика с замороженным актёром; "
               "в игре учится всё (актёр - с шагом в PLAY_LR раз меньше)",
        "honest_training_time": {
            "es_minutes": round(es_seconds / 60, 1),
            "es_generations": gen,
            "es_generation_chosen": chosen,
            "es_episodes": gen * es.PAIRS * 2 * 3,
            "es_lived_hours": round(gen * es.PAIRS * 2 * 3 * es.EPISODE / 3600, 2),
            "online_minutes": round(online_seconds / 60, 1),
            "online_worlds": worlds,
            "brain_steps": int(brain.steps),
            "lived_hours_online": round(brain.steps / STEPS_PER_SECOND / 3600, 2),
            "wall_minutes": round((es_seconds + online_seconds) / 60, 1),
        },
        "es": {"pairs": es.PAIRS, "sigma": es.SIGMA_ES, "lr": es.LR_ES, "episode_s": es.EPISODE,
               "jump_cost_m": es.JUMP_COST, "jitter_cost": es.JITTER_COST, "hollow_bonus_m": es.HOLLOW_BONUS,
               "started_from": args.init_note or ("случайные веса" if not args.init else args.init)},
        "snapshots": snapshots,
        "curve": curve[::max(1, len(curve) // 80)],
        "hyper": {"hidden": HIDDEN, "gamma": GAMMA, "sigma": SIGMA, "play_noise": PLAY_NOISE, "jerk": JERK,
                  "lr": LR},
        "check_10_new_worlds_60s_as_in_game": {"before": before, "after": after},
        "check_worst_hollow_30s": pits,
        "online_history": history[::max(1, len(history) // 40)],
    })
    print(f"готово за {(es_seconds + online_seconds) / 60:.1f} мин: поколений {gen}, "
          f"онлайн миров {worlds}, шагов {brain.steps}")


if __name__ == "__main__":
    main()
