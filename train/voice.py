"""Обучение голоса при сборке: сеть повторяет учителя голоса по состоянию мозга.

    python -m train.voice

Состояния - как у речи (train/speech.py: настоящая жизнь плюс редкие случаи), цели -
параметры синтезатора от учителя (teacher/voice.py) с небольшим шумом. Средний квадрат
ошибки, Adam. Проверка на новых состояниях - в models/voice.json.
"""
import time

from train.common import limit_threads, model_file, write_passport

limit_threads()

import numpy as np  # noqa: E402

from brain.state import IDX, STATE_DIM  # noqa: E402
from teacher import voice as teacher_voice  # noqa: E402
from train.speech import collect_states  # noqa: E402
from voice import synth  # noqa: E402
from voice.net import VoiceNet  # noqa: E402

STEPS = 4000


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    states = collect_states(6000, seed=1)
    rng.shuffle(states)
    held, train_s = states[:500], states[500:]
    targets = np.stack([teacher_voice.target(s, rng) for s in train_s])
    net = VoiceNet(seed=0)
    curve = []
    for step in range(STEPS):
        idx = rng.integers(0, len(train_s), 256)
        loss = net.train_batch(train_s[idx], targets[idx], lr=3e-3 * (0.1 ** (step / STEPS)))
        if step % 500 == 0 or step == STEPS - 1:
            curve.append({"step": step + 1, "loss": round(float(loss), 5)})
    clean = np.stack([teacher_voice.target(s, rng, noise=0.0) for s in held])
    pred = np.stack([net.params_for(s) for s in held])
    mse = float(((pred - clean) ** 2).mean())

    def st(**kw):
        s = np.zeros(STATE_DIM, np.float32)
        s[IDX["grounded"]] = 1
        s[IDX["curiosity"]] = 0.4
        for k, v in kw.items():
            s[IDX[k]] = v
        return s

    cases = {"упал": st(fell=1.0), "любопытно": st(curiosity=0.9), "прыжок": st(airborne=1.0, grounded=0),
             "скучно": st(curiosity=0.05)}
    by_case = {}
    for name, s in cases.items():
        p = net.params_for(s)
        by_case[name] = {"params": {k: round(float(v), 3) for k, v in zip(synth.PARAMS, p)},
                         "notes_hz": [round(f, 1) for f in synth.note_freqs(p)],
                         "seconds": round(synth.length_samples(p) / synth.RATE, 3)}
    net.save(model_file("voice.npz"))
    write_passport("voice", {
        "what": "плотная сеть 27 -> 24 -> 7: сводка состояния мозга -> параметры синтезатора",
        "teacher": "teacher/voice.py - только при обучении",
        "params": list(synth.PARAMS),
        "states": len(train_s), "steps": STEPS, "curve": curve,
        "check_new_states_mse": round(mse, 5),
        "by_state": by_case,
        "seconds": round(time.time() - t0, 1),
    })
    print(f"голос: ошибка на новых состояниях {mse:.4f}")


if __name__ == "__main__":
    main()
