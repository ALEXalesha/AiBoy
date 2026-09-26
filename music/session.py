"""Гитарист: одна попытка целиком - что учить, сыграть фразу, руки, какие ноты прозвучали,
награда, шаг обучения, запись в журнал.

Что учить (решение сверх спеки, записано в плане): каждая четвёртая попытка - свободная
игра (только B и C). Иначе - новейшая своя мелодия владельца, которая ещё не выучена, а если
таких нет - первая невыученная из песенника по порядку. Выучена фраза - лучшая похожесть A
не ниже LEARNED; во фразах мелодия учится по порядку. Если на мелодию ушло больше
GIVE_UP попыток, он переходит к следующей (и вернётся к ней, когда пройдёт по кругу).

Руки. Ноту на шаг t музыкант решает на шаг раньше: команда рукам - в начале шага t-1, и за
шаг суставы доезжают (тот же сглаживающий фильтр, что в физике тела). В режиме «Музыкант»
руки - анимация выбранных нот, звучит всё. В режиме «Руками» нота звучит, только если в
момент удара правая кисть у струны и палец левой в ладу: левая мимо - приглушённый тихий
звук, правая мимо - ничего. Промах в награде не слышен.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from body import physics, skeleton
from music import guitar, hands, reward, songbook
from music.guitar import Note
from music.journal import played
from music.musician import notes_of
from music.reward import Played
from music.taste import features

FREE_TEMPO = 100
FREE_STEPS = 32
FREE_EVERY = 4
LEARNED = 0.85
GIVE_UP = 300
LEAD = 2                     # шагов сетки: столько руке на дорогу до ноты
FPS = 60
L_SH, L_EL = skeleton.JOINTS.index("sh_l"), skeleton.JOINTS.index("el_l")
R_SH, R_EL = skeleton.JOINTS.index("sh_r"), skeleton.JOINTS.index("el_r")


@dataclass
class Attempt:
    tune: object                 # название мелодии или None - свободная игра
    phrase: object               # songbook.Phrase или None
    steps: int
    meter: int
    tempo: int
    take: object
    intended: list               # [(шаг, струна, лад, сила, длительность)]
    sounded: list                # 1 - прозвучала, 2 - приглушённо, 0 - нет
    arms: np.ndarray             # (кадров, 4): плечо и локоть левой, плечо и локоть правой
    motor: list = field(default_factory=list)   # расстояния кистей до целей в момент удара

    @property
    def step_sec(self):
        return 60.0 / self.tempo / 2.0

    @property
    def duration(self):
        return self.steps * self.step_sec

    def notes(self):
        """Звучащие ноты для синтеза (секунды)."""
        out = []
        for (t, s, f, force, d), snd in zip(self.intended, self.sounded):
            if snd == 1:
                out.append(Note(s, f, t * self.step_sec, force, d * self.step_sec))
            elif snd == 2:
                out.append(Note(s, 0, t * self.step_sec, 0, min(d, 1) * self.step_sec, muted=True))
        return out

    def played(self):
        return [Played(t, guitar.midi(s, f), d, force)
                for (t, s, f, force, d), snd in zip(self.intended, self.sounded) if snd == 1]


class Planned:
    """Руки для анимации в режиме «Музыкант»: углы до цели считаются геометрией."""

    def aim(self, body, lean, target, arm):
        return hands.reach(body, lean, target)

    def learn(self, attempt, body):
        pass


def arm_track(body, commands, frames, dt=1.0 / FPS):
    """Углы рук по кадрам: команды [(кадр, (плечо и локоть левой, правой))] проходят тот же
    двухступенчатый сглаживающий фильтр, что суставы тела в физике."""
    rest = np.array(hands.SITTING, float)
    angle = np.array([rest[L_SH], rest[L_EL], rest[R_SH], rest[R_EL]])
    command = angle.copy()
    target = angle.copy()
    out = np.zeros((frames, 4))
    k_cmd = 1.0 - math.exp(-dt / physics.TAU_CMD)
    k = 1.0 - math.exp(-dt / physics.TAU)
    cmd_i = 0
    commands = sorted(commands, key=lambda c: c[0])
    for fr in range(frames):
        while cmd_i < len(commands) and commands[cmd_i][0] <= fr:
            target = np.array(commands[cmd_i][1], float)
            cmd_i += 1
        command += (target - command) * k_cmd
        angle += np.clip((command - angle) * k, -physics.MAX_SPEED * dt, physics.MAX_SPEED * dt)
        out[fr] = angle
    return out


class Guitarist:
    def __init__(self, musician, journal, book, mode="musician", reacher=None, rng=None):
        self.musician = musician
        self.journal = journal
        self.book = book
        self.mode = mode
        self.reacher = reacher
        self.rng = rng or np.random.default_rng()
        self.taste = journal.taste()
        self.count = 0
        self.last = None             # (попытка, запись журнала) - для оценки после конца

    # --- что играть ---
    def _best(self, tune, phrase):
        vals = [e["A"] for e in self.journal.entries
                if e.get("tune") == tune and e.get("phrase") == phrase and e.get("A") is not None]
        return max(vals) if vals else 0.0

    def _tries(self, tune):
        return sum(1 for e in self.journal.entries if e.get("tune") == tune)

    def target(self):
        """(название, фраза) или (None, None) - свободная игра."""
        self.count += 1
        if self.count % FREE_EVERY == 0:
            return None, None
        tunes = list(reversed(self.book.own)) + list(songbook.BUILTIN)
        fresh = [t for t in tunes if self._tries(t.name) < GIVE_UP] or tunes
        for t in fresh:
            for ph in songbook.phrases(t):
                if self._best(t.name, ph.index) < LEARNED:
                    return t.name, ph
        return None, None

    # --- попытка ---
    def attempt(self, body, lean=None):
        lean = hands.SITTING[0] if lean is None else lean
        tune, phrase = self.target()
        if phrase is None:
            steps, meter, tempo, target = FREE_STEPS, 8, FREE_TEMPO, None
        else:
            steps, meter, tempo, target = phrase.steps, phrase.meter, phrase.tempo, phrase.notes
        take = self.musician.play(steps, meter, target)
        intended = notes_of(take.actions, steps)
        step_sec = 60.0 / tempo / 2.0
        frames = int(math.ceil(steps * step_sec * FPS)) + 1
        reacher = self.reacher if (self.mode == "hands" and self.reacher is not None) else Planned()
        if hasattr(reacher, "trace"):
            reacher.trace = {k: [] for k in reacher.trace}
            reacher.prev = {k: None for k in reacher.prev}
        commands = []
        aims = []
        for t, s, f, force, d in intended:
            lt = hands.left_target(s, f, body)
            rt = hands.right_target(s, body)
            left = reacher.aim(body, lean, lt, "left")
            right = reacher.aim(body, lean, rt, "right")
            start = int(max(0, t - LEAD) * step_sec * FPS)
            commands.append((start, (left[0], left[1], right[0], right[1])))
            # после удара правая рука чуть отходит от струн
            commands.append((int((t + 0.5) * step_sec * FPS), (left[0], left[1], right[0] - 0.15, right[1])))
            aims.append((t, s, f, lt, rt, left, right))
        arms = arm_track(body, commands, frames)
        sounded, motor, reached = [], [], []
        for (t, s, f, lt, rt, _, _) in aims:
            fr = min(frames - 1, int(t * step_sec * FPS))
            lp = hands.hand(body, lean, arms[fr, 0], arms[fr, 1])
            rp = hands.hand(body, lean, arms[fr, 2], arms[fr, 3])
            motor.append((math.dist(lp, lt), math.dist(rp, rt)))
            reached.append((lp, rp))
            if self.mode != "hands":
                sounded.append(1)
            elif not hands.right_hit(rp, s, body):
                sounded.append(0)
            elif not hands.left_hit(lp, s, f, body):
                sounded.append(2)
            else:
                sounded.append(1)
        a = Attempt(tune, phrase, steps, meter, tempo, take, intended, sounded, arms, motor)
        a.aims = aims
        a.reached = reached
        a.lean = lean
        return a

    # --- после попытки ---
    def history(self):
        return [played(e) for e in self.journal.entries[-reward.HISTORY:]]

    def finish(self, attempt, body=None, rating=None):
        """Награда, шаг обучения музыканта (и рук в режиме «Руками»), запись в журнал."""
        notes = attempt.played()
        b = reward.pleasant(notes, attempt.meter, self.history(), attempt.steps)
        a = None if attempt.phrase is None else reward.closeness(notes, attempt.phrase.notes)
        feats = features(notes, a, attempt.steps, attempt.meter)
        guess = self.taste.predict(feats)
        c = float(rating) if rating is not None else (guess if self.taste.net is not None else None)
        total = reward.total(a, b.total, c)
        key = f"{attempt.tune}/{attempt.phrase.index}" if attempt.phrase is not None else "free"
        before = self.musician.baseline.get(key, total)
        self.musician.learn(attempt.take, total, key)
        if self.mode == "hands" and self.reacher is not None and body is not None:
            self.reacher.learn(attempt, body)
        entry = self.journal.add({
            "tune": attempt.tune, "phrase": None if attempt.phrase is None else attempt.phrase.index,
            "mode": self.mode, "tempo": attempt.tempo, "meter": attempt.meter, "steps": attempt.steps,
            "notes": [[t, s, f, force, d, snd] for (t, s, f, force, d), snd in zip(attempt.intended, attempt.sounded)],
            "A": a, "B": b.total, "C": None if rating is None else float(rating),
            "taste": guess if self.taste.net is not None else None, "total": total, "rating": rating,
            "novelty": b.novelty, "motor": (float(np.mean([m[0] + m[1] for m in attempt.motor]) / 2)
                                             if attempt.motor else None)})
        if rating is not None:
            self._taste_learn(feats, rating)
        self.last = (attempt, entry, feats, key)
        progress = float(np.clip(0.5 + 2.0 * (total - before), 0.0, 1.0))
        return entry, progress, b.novelty

    def _taste_learn(self, feats, rating):
        self.taste.rate(feats, rating)
        self.journal.guesses, self.journal.hits = self.taste.guesses, self.taste.hits

    def rate_last(self, rating):
        """👍/👎 после конца попытки: запись в журнал, вкус учится, и ещё один шаг музыканта
        на той же попытке - с оценкой владельца вместо догадки."""
        if self.last is None:
            return None
        attempt, entry, feats, key = self.last
        if entry.get("rating") == rating:
            return entry
        old_c = entry.get("C") if entry.get("C") is not None else (entry.get("taste") or 0.0)
        self.journal.rate(entry["id"], rating)
        entry["C"] = float(rating)
        entry["total"] = entry["total"] + reward.WEIGHTS["C"] * (rating - old_c)
        self.musician.learn(attempt.take, entry["total"], key)
        self._taste_learn(feats, rating)
        return entry
