"""Замер: сколько длится сеанс гитары и что делают интересы к миру и к музыке.

    python tools\\measure_sessions.py [--minutes 30] [--freq sometimes] [--bodies 5]

Как в игре (обучение на ходу), без окна: у каждого человечка гитара уже за спиной, он
живёт minutes минут игрового времени. Печатает сеансы (начало, длина, попыток), время
между ними и ход интересов (мир / музыка) по минутам.
"""
import argparse
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np  # noqa: E402

from game import models  # noqa: E402
from game.life import Life  # noqa: E402
from music.journal import Journal  # noqa: E402
from music.musician import Musician  # noqa: E402
from music.session import Guitarist  # noqa: E402
from music.songbook import Songbook  # noqa: E402

BODIES = (7, 21, 34, 55, 89)
FLICKER = 15.0          # с


def sessions(body_seed, minutes, freq="sometimes", world_seed=None, width=160, log=None, chest=None):
    """chest - какие игрушки в сундуке (None - все; законы сеансов гитары - одна гитара: с 2.0
    в сундуке ещё 8 игрушек, и гитаре за полчаса достаётся один сеанс или ни одного)."""
    d = tempfile.mkdtemp(prefix="aiboy-sessions-")
    g = Guitarist(Musician(seed=body_seed), Journal(os.path.join(d, "j.json")), Songbook(os.path.join(d, "s.json")),
                  rng=np.random.default_rng(body_seed))
    m = models.load()
    life = Life(m, world_seed or 1000 + body_seed, body_seed, width, models.starter_brain(), guitarist=g,
                has_guitar=True, guitar_freq=freq, chest=chest)
    out, cur, trace, phrases = [], None, [], []
    life.laid = []             # (время, где положил, где потом взял или None)
    for i in range(int(minutes * 3600)):
        events = life.tick()
        for kind, *data in events:
            if kind == "sit":
                cur = {"start": life.time, "attempts": 0}
            elif kind == "attempt" and cur is not None:
                cur["attempts"] += 1
            elif kind == "stand" and cur is not None:
                cur["length"] = life.time - cur["start"]
                out.append(cur)
                cur = None
            elif kind == "put":
                if out:
                    out[-1]["put"] = True
                life.laid.append([life.time, life.guitar_x, None])
            elif kind == "pickup" and life.laid and life.laid[-1][2] is None:
                life.laid[-1][2] = life.human.px
            elif kind == "phrase":
                phrases.append((round(life.time), data[0], life.playing))
        if i % 3600 == 0:
            c = life.choice
            trace.append((round(life.time / 60), round(c.acts["explore"].pull(), 2), round(c.acts["guitar"].pull(), 2),
                          "играет" if life.playing else "гуляет"))
    if cur is not None:
        cur["length"] = life.time - cur["start"]
        cur["unfinished"] = True
        out.append(cur)
    return out, trace, phrases, life.laid


def all_sessions(body_seed, minutes, world_seed=None, width=160, dog=True):
    """Все сеансы игр человечка за minutes минут (гитара, игрушки, плавание, собака, гора,
    маска): [(занятие, длина, с)]. Сеанс - пока выбор занятий держит это занятие подряд;
    незаконченный в конце - с пометкой (длина отрицательная - не в счёт медианы, но в «не дольше»)."""
    from toys import registry
    play = {t.name for t in registry.TOYS.values() if not t.tool} | {"swim", "dog", "climb", "dive"}
    d = tempfile.mkdtemp(prefix="aiboy-all-")
    g = Guitarist(Musician(seed=body_seed), Journal(os.path.join(d, "j.json")), Songbook(os.path.join(d, "s.json")),
                  rng=np.random.default_rng(body_seed))
    m = models.load()
    life = Life(m, world_seed or 1000 + body_seed, body_seed, width, models.starter_brain(), guitarist=g,
                guitar_freq="sometimes", dog_mode="often" if dog else "never")
    if dog and life.dog is None:
        life.spawn_dog(near=True)
    out, cur, start = [], None, 0.0
    for _ in range(int(minutes * 3600)):
        life.tick()
        now = life.choice.current
        if now != cur:
            # меньше FLICKER - не сеанс: выбрал гитару, а она в сундуке - выбор на миг уходит в «гулять»,
            # пока он идёт за ней (так устроено с 1.1.0), сеанс начнётся, когда сядет играть
            if cur in play and life.time - start >= FLICKER:
                out.append((cur, life.time - start))
            cur, start = now, life.time
    if cur in play:
        out.append((cur, -(life.time - start)))    # не кончился к концу замера
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--freq", default="sometimes")
    ap.add_argument("--bodies", type=int, default=5)
    args = ap.parse_args()
    t0 = time.perf_counter()
    lengths = []
    for b in BODIES[:args.bodies]:
        out, trace, _, laid = sessions(b, args.minutes, args.freq)
        lengths += [s["length"] / 60 for s in out]
        print(f"человечек {b}: сеансов {len(out)}: " + ", ".join(
            f"{s['start'] / 60:.1f} мин +{s['length'] / 60:.1f} мин ({s['attempts']} поп.)"
            + (" не кончился" if s.get("unfinished") else "") for s in out), flush=True)
        print("   положил:", [(round(t / 60, 1), round(x, 1), None if y is None else round(y, 1)) for t, x, y in laid],
              "; мин/мир/гитара:", trace, f"{time.perf_counter() - t0:.0f} с", flush=True)
    if lengths:
        print(f"медиана сеанса {np.median(lengths):.1f} мин, самый долгий {max(lengths):.1f} мин, всего {len(lengths)}")


if __name__ == "__main__":
    main()
