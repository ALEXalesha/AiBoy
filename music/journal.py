"""Журнал попыток человечка: каждая попытка - ноты, мелодия, части награды, оценка
владельца, время. Звук не хранится: при «Слушать» он синтезируется заново - места мало.

Не больше MAX_ENTRIES на человечка; сверх - остаются первые KEEP_FIRST, каждая десятая (по
номеру) и последние KEEP_LAST: видно, с чего начинал, как шёл и где сейчас.
"""
import math
import time

from game.storage import is_int, load_json, save_json
from music.reward import Played
from music.taste import Taste, features

MAX_ENTRIES = 2000
KEEP_FIRST = 20
KEEP_LAST = 200


def _num_or_none(v):
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))


def _ok(e):
    if not isinstance(e, dict) or not is_int(e.get("id")) or e["id"] < 1:
        return False
    notes = e.get("notes")
    if not isinstance(notes, list) or not all(isinstance(n, list) and len(n) == 6 and all(is_int(v) for v in n)
                                              for n in notes):
        return False
    if not (e.get("tune") is None or isinstance(e.get("tune"), str)):
        return False
    if e.get("rating") not in (None, 1, -1):
        return False
    return all(_num_or_none(e.get(k)) for k in ("A", "B", "C", "taste", "total"))


def played(entry):
    """Прозвучавшие ноты попытки в шагах сетки (для награды и вкуса)."""
    from music import guitar
    return [Played(t, guitar.midi(s, f), d, force) for t, s, f, force, d, sounded in entry["notes"] if sounded]


class Journal:
    def __init__(self, path, entries=None, next_id=1, guesses=0, hits=0, meta=None):
        self.path = path
        self.meta = dict(meta or {})         # есть ли гитара, настроение и прочее о человечке
        self.entries = list(entries or [])
        self.next_id = next_id
        self.guesses, self.hits = guesses, hits

    @classmethod
    def load(cls, path):
        raw = load_json(path)
        if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
            return cls(path)
        entries = [e for e in raw["entries"] if _ok(e)]
        nxt = raw.get("next_id")
        nxt = nxt if is_int(nxt) and nxt > max([e["id"] for e in entries] or [0]) else \
            max([e["id"] for e in entries] or [0]) + 1
        g, h = raw.get("guesses", 0), raw.get("hits", 0)
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        return cls(path, entries, nxt, g if is_int(g) else 0, h if is_int(h) else 0, meta)

    def save(self):
        return save_json(self.path, {"next_id": self.next_id, "guesses": self.guesses, "hits": self.hits,
                                     "meta": self.meta, "entries": self.entries})

    def add(self, entry):
        e = dict(entry)
        e["id"] = self.next_id
        e.setdefault("time", time.time())
        self.next_id += 1
        self.entries.append(e)
        self.thin()
        return e

    def thin(self):
        n = len(self.entries)
        if n <= MAX_ENTRIES:
            return
        self.entries = [e for i, e in enumerate(self.entries)
                        if i < KEEP_FIRST or i >= n - KEEP_LAST or e["id"] % 10 == 0]

    def get(self, entry_id):
        return next((e for e in self.entries if e["id"] == entry_id), None)

    def rate(self, entry_id, rating):
        e = self.get(entry_id)
        if e is not None:
            e["rating"] = rating
        return e

    def first_of(self, tune):
        return next((e for e in self.entries if e.get("tune") == tune), None)

    def progress(self, tune):
        """[(номер попытки, оценка)] по мелодии - похожесть A, в свободной игре - B."""
        key = "B" if tune is None else "A"
        return [(e["id"], e[key]) for e in self.entries if e.get("tune") == tune and e.get(key) is not None]

    def best(self, tune):
        vals = [v for _, v in self.progress(tune)]
        return max(vals) if vals else None

    def tunes(self):
        seen = []
        for e in self.entries:
            if e.get("tune") not in seen:
                seen.append(e.get("tune"))
        return seen

    def taste(self):
        ratings = [(features(played(e), e.get("A"), e.get("steps", 32), e.get("meter", 8)), e["rating"])
                   for e in self.entries if e.get("rating") in (1, -1)]
        return Taste.from_ratings(ratings, self.guesses, self.hits)
