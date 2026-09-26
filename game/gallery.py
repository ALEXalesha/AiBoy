"""Галерея: миры с человечками, которые уже были. Пополняется сама - каждый новый мир
попадает сюда; хранятся только зёрна и итоги жизни, картинка рисуется заново. Новые -
первыми, не больше MAX_ENTRIES. Удаление - из данных игры, без Корзины."""
import math
import time

from game.storage import is_int, load_json, save_json

MAX_ENTRIES = 60
REQUIRED = {"id": str, "world_seed": int, "body_seed": int, "width": float, "world": str, "human": str}
NUMBERS = ("lived", "distance", "phrases", "interest", "created")


def make_entry(world_seed, body_seed, width, world, human, interest, created=None):
    return {"id": f"{int(world_seed)}-{int(body_seed)}-{int(width)}", "world_seed": int(world_seed),
            "body_seed": int(body_seed), "width": float(width), "world": world, "human": human,
            "interest": float(interest), "created": float(created if created is not None else time.time()),
            "lived": 0.0, "distance": 0.0, "phrases": 0}


def _ok(e):
    if not isinstance(e, dict):
        return False
    for key, kind in REQUIRED.items():
        v = e.get(key)
        if kind is int and not is_int(v):
            return False
        if kind is float and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                  and v > 0):
            return False
        if kind is str and not (isinstance(v, str) and v):
            return False
    for key in NUMBERS:
        v = e.get(key, 0)
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0):
            return False
    return True


class Gallery:
    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def get(self, entry_id):
        return next((e for e in self.entries if e["id"] == entry_id), None)

    def add(self, entry):
        self.entries = [e for e in self.entries if e["id"] != entry["id"]]
        self.entries.insert(0, dict(entry))
        del self.entries[MAX_ENTRIES:]
        return self.entries[0]

    def update(self, entry_id, **fields):
        e = self.get(entry_id)
        if e is not None:
            e.update(fields)
        return e

    def remove(self, entry_id):
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        return len(self.entries) < before

    def most_interesting(self, n=5):
        return sorted(self.entries, key=lambda e: -e.get("interest", 0))[:n]

    @classmethod
    def load(cls, path):
        raw = load_json(path)
        if not isinstance(raw, list):
            return cls()
        seen, out = set(), []
        for e in raw:
            if _ok(e) and e["id"] not in seen:
                seen.add(e["id"])
                out.append({**{k: 0 for k in NUMBERS}, **e, "width": float(e["width"])})
        return cls(out[:MAX_ENTRIES])

    def save(self, path):
        return save_json(path, self.entries)
