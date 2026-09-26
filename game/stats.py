"""Статистика за всё время: миры, прожитое время, путь, фразы, прыжки, падения,
частые слова и кривая обучения мозга (средняя награда любопытства по прожитому времени);
гитара - время с гитарой, попытки, оценки 👍/👎, лучшая похожесть по мелодии.
Битый файл - пустая статистика, битое поле - ноль."""
import math

from game.storage import is_int, load_json, save_json

COUNTS = ("worlds", "phrases", "jumps", "falls", "attempts", "likes", "dislikes")
AMOUNTS = ("lived", "distance", "guitar_time")
MAX_CURVE = 400
MAX_WORDS = 300


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def words_of(text):
    return [w for w in "".join(c if c.isalpha() or c == "-" else " " for c in text.lower()).split()
            if len(w.strip("-")) >= 3]


class Stats:
    def __init__(self):
        self.worlds = self.phrases = self.jumps = self.falls = 0
        self.attempts = self.likes = self.dislikes = 0
        self.lived = self.distance = self.guitar_time = 0.0
        self.best = {}             # мелодия -> лучшая похожесть A
        self.words = {}
        self.curve = []            # [[прожито секунд, средняя награда]]

    def add_world(self):
        self.worlds += 1

    def add_life(self, seconds, meters):
        self.lived += seconds
        self.distance += meters

    def add_phrase(self, text):
        self.phrases += 1
        for w in words_of(text):
            self.words[w] = self.words.get(w, 0) + 1
        if len(self.words) > MAX_WORDS:
            keep = sorted(self.words.items(), key=lambda kv: -kv[1])[:MAX_WORDS // 2]
            self.words = dict(keep)

    def add_event(self, name):
        if name == "jump":
            self.jumps += 1
        elif name == "fall":
            self.falls += 1

    def add_curve(self, lived, value):
        self.curve.append([round(float(lived), 1), round(float(value), 5)])
        if len(self.curve) > MAX_CURVE:
            # соседние точки попарно в одну: кривая целиком, но вдвое реже
            merged = [[b[0], (a[1] + b[1]) / 2] for a, b in zip(self.curve[0::2], self.curve[1::2])]
            if len(self.curve) % 2:
                merged.append(self.curve[-1])
            self.curve = merged

    def add_attempt(self, entry, best=None):
        self.attempts += 1
        tune = entry.get("tune")
        if tune is not None and entry.get("A") is not None:
            self.best[tune] = max(self.best.get(tune, 0.0), float(entry["A"]))
        if entry.get("rating") is not None:
            self.guitar_rating(entry["rating"])

    def guitar_rating(self, rating):
        if rating == 1:
            self.likes += 1
        elif rating == -1:
            self.dislikes += 1

    def top_words(self, n=12):
        return sorted(self.words.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def to_dict(self):
        d = {k: getattr(self, k) for k in COUNTS + AMOUNTS}
        d["words"] = dict(self.words)
        d["curve"] = [list(p) for p in self.curve]
        d["best"] = dict(self.best)
        return d

    @classmethod
    def load(cls, path):
        raw = load_json(path)
        s = cls()
        if not isinstance(raw, dict):
            return s
        for k in COUNTS:
            if is_int(raw.get(k)) and raw[k] >= 0:
                setattr(s, k, raw[k])
        for k in AMOUNTS:
            if _num(raw.get(k)):
                setattr(s, k, float(raw[k]))
        if isinstance(raw.get("words"), dict):
            s.words = {w: c for w, c in raw["words"].items() if isinstance(w, str) and is_int(c) and c > 0}
        if isinstance(raw.get("best"), dict):
            s.best = {k: float(v) for k, v in raw["best"].items()
                      if isinstance(k, str) and _num(v) and v <= 1.0}
        if isinstance(raw.get("curve"), list):
            s.curve = [[float(p[0]), float(p[1])] for p in raw["curve"]
                       if isinstance(p, list) and len(p) == 2 and _num(p[0])
                       and isinstance(p[1], (int, float)) and not isinstance(p[1], bool) and math.isfinite(p[1])]
        return s

    def save(self, path):
        return save_json(path, self.to_dict())
