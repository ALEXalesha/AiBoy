"""Настройки. Сохраняются сразу при изменении; каждое плохое поле в файле заменяется
значением по умолчанию само по себе, остальные остаются."""
from dataclasses import asdict, dataclass, fields

from game.storage import is_int, load_json, save_json

THEMES = ("dark", "light")
FREQS = ("rare", "normal", "often")
FREQ_NAMES = {"rare": "Редко", "normal": "Обычно", "often": "Часто"}
SPEEDS = (1, 2, 4)
SIZES = ("small", "medium", "large")
SIZE_NAMES = {"small": "Маленький (80 м)", "medium": "Средний (160 м)", "large": "Большой (320 м)"}
WORLD_WIDTH = {"small": 80.0, "medium": 160.0, "large": 320.0}


@dataclass
class Settings:
    volume: int = 60             # 0..100
    phrase_freq: str = "normal"
    speed: int = 1               # скорость по умолчанию: 1, 2, 4
    show_thoughts: bool = True
    theme: str = "dark"
    world_size: str = "medium"

    @staticmethod
    def valid(name, value):
        checks = {
            "volume": lambda v: is_int(v) and 0 <= v <= 100,
            "phrase_freq": lambda v: v in FREQS,
            "speed": lambda v: is_int(v) and v in SPEEDS,
            "show_thoughts": lambda v: isinstance(v, bool),
            "theme": lambda v: v in THEMES,
            "world_size": lambda v: v in SIZES,
        }
        try:
            return checks[name](value)
        except TypeError:           # нехэшируемое значение в проверке «in»
            return False

    @classmethod
    def load(cls, path):
        data = load_json(path)
        s = cls()
        if isinstance(data, dict):
            for f in fields(cls):
                if f.name in data and cls.valid(f.name, data[f.name]):
                    setattr(s, f.name, data[f.name])
        return s

    def save(self, path):
        return save_json(path, asdict(self))
