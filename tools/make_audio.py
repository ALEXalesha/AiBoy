"""Примеры звука для README: docs/audio/*.wav - как музыкант учит «Чижика-пыжика».

    python tools\\make_audio.py [--seed 1] [--middle 300] [--max 3000]

Музыкант с нуля учит первую фразу (4 такта) без владельца: награда A + B, как в игре без
оценок. Сохраняются три попытки: самая первая, попытка номер --middle (примерно столько
нужно, чтобы стало слышно по-другому) и первая попытка, в которой фраза выучена
(похожесть A не ниже LEARNED), а если за --max попыток такой не нашлось - лучшая. В конце
к каждой попытке добавлена сама мелодия, сыгранная точно по нотам (после паузы) - чтобы
было с чем сравнить на слух. Печатает для каждой: A, пик (перегруза нет - меньше 1),
длительность.
"""
import argparse
import os
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np  # noqa: E402

from music import guitar, reward, songbook  # noqa: E402
from music import musician as mus  # noqa: E402
from music.guitar import Note  # noqa: E402
from music.session import LEARNED  # noqa: E402
from music.string import SR, StringBank  # noqa: E402
from tools.measure_learning import played_of  # noqa: E402

TUNE = "Чижик-пыжик"
OUT = os.path.join(ROOT, "docs", "audio")
GAP = 1.0            # с тишины между попыткой и мелодией
RATE = 22050


def take_notes(take, step):
    return [Note(s, f, t * step, force, d * step) for t, s, f, force, d in mus.notes_of(take.actions, take.steps)]


def tune_notes(phrase, step):
    """Мелодия фразы точно по нотам: для каждой высоты - первое место на грифе."""
    out = []
    for n in phrase.notes:
        s, f = guitar.positions(n.pitch)[0]
        out.append(Note(s, f, n.onset * step, 2, n.dur * step))
    return out


def write_wav(path, samples):
    """16 бит, моно, RATE Гц: вдвое меньше файл, струна выше 11 кГц почти не звучит."""
    spec = np.fft.rfft(samples.astype(np.float64))
    n = len(samples) * RATE // SR
    data = np.fft.irfft(spec[:n // 2 + 1], n) * (RATE / SR)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--middle", type=int, default=300)
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--dry", action="store_true", help="только учить и печатать, без файлов")
    args = ap.parse_args()
    tune = songbook.by_name(TUNE)
    ph = songbook.phrases(tune)[0]
    step = 60.0 / ph.tempo / 2.0
    length = ph.steps * step
    m = mus.Musician(seed=args.seed)
    history = []
    keep = {}
    best = (-1.0, None, 0)
    started = time.perf_counter()
    for i in range(1, args.max + 1):
        take = m.play(ph.steps, ph.meter, ph.notes)
        played = played_of(take)
        a = reward.closeness(played, ph.notes)
        b = reward.pleasant(played, ph.meter, history, ph.steps).total
        history = (history + [played])[-reward.HISTORY:]
        m.learn(take, reward.total(a, b, None), key=f"{TUNE}/0")
        if i == 1:
            keep["first"] = (take, a, i)
        if i == args.middle:
            keep["middle"] = (take, a, i)
        if a > best[0]:
            best = (a, take, i)
        if a >= LEARNED and i > args.middle:
            keep["learned"] = (take, a, i)
            break
    if "learned" not in keep:
        # не выучил случайной игрой - как он играет «всерьёз», без случайности
        take = m.play(ph.steps, ph.meter, ph.notes, greedy=True)
        a = reward.closeness(played_of(take), ph.notes)
        keep["learned"] = (take, a, i) if a >= best[0] else (best[1], best[0], best[2])
    print(f"учил {i} попыток, {time.perf_counter() - started:.0f} с; выучено - попытка "
          f"{keep['learned'][2]}, A = {keep['learned'][1]:.2f}")
    if args.dry:
        return
    os.makedirs(OUT, exist_ok=True)
    bank = StringBank()
    melody = bank.render(tune_notes(ph, step), length)
    names = {"first": "chizhik-01-first", "middle": f"chizhik-02-after-{args.middle}",
             "learned": "chizhik-03-learned"}
    for key, (take, a, n) in keep.items():
        mine = bank.render(take_notes(take, step), length)
        full = np.concatenate([mine, np.zeros(int(GAP * SR), np.float32), melody])
        path = os.path.join(OUT, names[key] + ".wav")
        write_wav(path, full)
        peak = float(np.max(np.abs(full)))
        print(f"{os.path.basename(path)}: попытка {n}, похоже A = {a:.2f}, пик {peak:.2f}, "
              f"{len(full) / SR:.1f} с, {os.path.getsize(path) // 1024} КБ")


if __name__ == "__main__":
    main()
