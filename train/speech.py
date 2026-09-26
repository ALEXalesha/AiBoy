"""Обучение речи при сборке: состояния из настоящей жизни, фразы - от учителя.

    python -m train.speech [--states 24000] [--epochs 10]

1. Состояния. Человечки с необученным мозгом живут в мирах сети мира (models/world.npz),
   каждые 3 шага мозга снимается сводка (brain/state.py). Часть состояний дополнена
   тем, что в коротком прогоне бывает редко: упал, летит, хочет прыгнуть, любопытство
   и направление - любые.
2. Фразы. На каждое состояние учитель (teacher/phrases.py) пишет две фразы.
3. Сеть (speech/net.py) учится предсказывать следующую букву фразы по состоянию.
4. Проверка на новых состояниях: доля фраз только из слов учителя, сколько разных фраз,
   различаются ли фразы у разных состояний. Всё - в models/speech.json.
"""
import argparse
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from body.cppn import generate_body  # noqa: E402
from body.physics import Human  # noqa: E402
from brain.brain import Brain, PHYSICS_PER_STEP  # noqa: E402
from brain.observe import observe  # noqa: E402
from brain.state import IDX, STATE_DIM, summarize  # noqa: E402
from game import models  # noqa: E402
from speech import vocab  # noqa: E402
from speech.net import HIDDEN, SpeechNet  # noqa: E402
from teacher import phrases  # noqa: E402

WORLD_WIDTH = 160.0
TEMPERATURE = 0.75


def collect_states(n, seed=0, log=print):
    m = models.load()
    rng = np.random.default_rng(seed)
    out = []
    world_i = 0
    while len(out) < n:
        from world.generate import generate
        world = generate(m.world_genome, 50_000 + seed * 1000 + world_i, WORLD_WIDTH)
        body = generate_body(m.body_genome, 70_000 + world_i)
        human = Human(body, world, x=float(rng.uniform(0, WORLD_WIDTH)))
        brain = Brain(seed=seed * 100 + world_i)
        fell = 0.0
        for step in range(900):
            d, _ = brain.step(observe(human, world), learn=False)
            events = []
            for k in range(PHYSICS_PER_STEP):
                events += human.step(d.targets, d.turn, d.jump and k == 0)
            fell = 4.0 if "fall" in events else max(0.0, fell - 2 / 60)
            if step % 3 == 0:
                out.append(summarize(human, world, d.turn, d.want_jump, brain.curiosity, fell / 4.0))
        world_i += 1
        if world_i % 10 == 0:
            log(f"  миров {world_i}, состояний {len(out)}")
    states = np.stack(out[:n])
    return augment(states, rng)


def augment(states, rng):
    s = states.copy()
    n = len(s)
    r = rng.random(n)
    s[r < 0.12, IDX["fell"]] = rng.uniform(0.5, 1.0, (r < 0.12).sum())
    air = (r >= 0.12) & (r < 0.2)
    s[air, IDX["airborne"]] = 1.0
    s[air, IDX["grounded"]] = 0.0
    k = rng.random(n) < 0.35
    s[k, IDX["want_jump"]] = rng.uniform(0, 0.7, k.sum())
    k = rng.random(n) < 0.5
    s[k, IDX["curiosity"]] = rng.uniform(0, 1, k.sum())
    k = rng.random(n) < 0.4
    s[k, IDX["want_dir"]] = rng.uniform(-1, 1, k.sum())
    k = rng.random(n) < 0.15
    s[k, IDX["night"]] = 1.0 - s[k, IDX["night"]]
    return s


def make_pairs(states, per_state, rng):
    xs, ts = [], []
    for s in states:
        for _ in range(per_state):
            xs.append(s)
            ts.append(phrases.phrase(s, rng))
    return np.stack(xs).astype(np.float32), ts


def evaluate(net, states, rng):
    words = phrases.vocabulary()
    samples = [net.sample(s, rng, TEMPERATURE) for s in states]
    ok = [bool(p) and set(phrases.words(p)) <= words for p in samples]
    greedy = [net.greedy(s) for s in states[:200]]
    pairs = [(i, j) for i in range(0, 200, 2) for j in (i + 1,)]
    differ = [greedy[i] != greedy[j] for i, j in pairs if np.abs(states[i] - states[j]).sum() > 2.0]
    return {
        "states": len(states),
        "teacher_words_rate": round(float(np.mean(ok)), 4),
        "distinct_phrases": len(set(samples)),
        "different_states_different_greedy_phrase": round(float(np.mean(differ)), 4) if differ else None,
        "examples": samples[:25],
    }


def named_examples(net, rng):
    def st(**kw):
        s = np.zeros(STATE_DIM, np.float32)
        s[IDX["grounded"]] = 1.0
        s[IDX["facing"]] = 1.0
        s[IDX["curiosity"]] = 0.4
        for k, v in kw.items():
            s[IDX[k]] = v
        return s
    blue_tree = st(kind_tree=1, entity=1, entity_side=1, entity_dist=0.1, hue_cos=np.cos(2 * np.pi * 0.62),
                   hue_sin=np.sin(2 * np.pi * 0.62), sat=0.8, val=0.9)
    cases = {"упал": st(fell=1.0), "холм справа": st(hill_right=1.0), "синее дерево рядом": blue_tree,
             "скучно": st(curiosity=0.05), "ночь": st(night=1.0, curiosity=0.4), "летит": st(airborne=1, grounded=0)}
    return {name: [net.sample(s, rng, TEMPERATURE) for _ in range(4)] for name, s in cases.items()}


def train(n_states, epochs, batch=128, seed=0, log=print):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    log("состояния...")
    states = collect_states(n_states, seed, log)
    rng.shuffle(states)
    held = states[:600]
    train_states = states[600:]
    xs, ts = make_pairs(train_states, 2, rng)
    log(f"пар для обучения: {len(ts)}, собрано за {time.time() - t0:.0f} с")
    net = SpeechNet(hidden=HIDDEN, seed=seed)
    curve = []
    steps_per_epoch = len(ts) // batch
    total = epochs * steps_per_epoch
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(ts))
        losses = []
        for b in range(steps_per_epoch):
            idx = order[b * batch:(b + 1) * batch]
            lr = 3e-3 * (0.1 ** (step / total))
            losses.append(net.train_batch(xs[idx], [ts[i] for i in idx], lr=lr))
            step += 1
        curve.append({"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4)})
        log(f"эпоха {ep + 1}: потеря {np.mean(losses):.3f}, {time.time() - t0:.0f} с")
    seconds = time.time() - t0
    report = evaluate(net, held, np.random.default_rng(seed + 1))
    return net, {"curve": curve, "seconds": seconds, "pairs": len(ts), "check": report,
                 "named": named_examples(net, np.random.default_rng(seed + 2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=24000)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    net, info = train(args.states, args.epochs)
    net.save(model_file("speech.npz"))
    write_passport("speech", {
        "what": "посимвольная GRU: сводка состояния мозга (27 чисел) -> фраза по-русски",
        "network": {"hidden": HIDDEN, "vocab": vocab.SIZE, "state": STATE_DIM, "max_len": vocab.MAX_LEN,
                    "parameters": int(sum(p.size for _, p, _ in net.params()))},
        "teacher": "teacher/phrases.py - только при обучении; в игре только сеть",
        "states": args.states, "pairs": info["pairs"], "epochs": args.epochs,
        "seconds": round(info["seconds"], 1), "curve": info["curve"],
        "temperature": TEMPERATURE,
        "check_new_states": info["check"],
        "examples_by_state": info["named"],
    })
    print(f"готово: {info['check']['teacher_words_rate']:.3f} фраз из слов учителя, "
          f"{info['check']['distinct_phrases']} разных")


if __name__ == "__main__":
    main()
