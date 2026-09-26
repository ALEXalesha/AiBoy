"""Настройки. Сохраняются сразу при изменении; каждое плохое поле в файле заменяется
значением по умолчанию само по себе, остальные остаются."""
from dataclasses import asdict, dataclass, fields

from game.storage import is_int, load_json, save_json

THEMES = ("dark", "light")
FREQS = ("rare", "normal", "often")
FREQ_NAMES = {"rare": "Редко", "normal": "Обычно", "often": "Часто"}
SPEEDS = (1, 2, 4, 8)
SIZES = ("small", "medium", "large")
SIZE_NAMES = {"small": "Маленький (80 м)", "medium": "Средний (160 м)", "large": "Большой (320 м)"}
WORLD_WIDTH = {"small": 80.0, "medium": 160.0, "large": 320.0}
GUITAR_MODES = ("musician", "hands")
GUITAR_MODE_NAMES = {"musician": "Музыкант", "hands": "Руками"}
GUITAR_FREQS = ("rare", "sometimes", "often")
GUITAR_FREQ_NAMES = {"rare": "Редко", "sometimes": "Иногда", "often": "Часто"}
GUITAR_FAST = ("slow", "silent")
GUITAR_FAST_NAMES = {"slow": "Жизнь идёт в x1 - слышно каждую попытку",
                     "silent": "Скорость не меняется, попытки на x2-x8 - без звука"}


@dataclass
class Settings:
    phrase_freq: str = "normal"
    speed: int = 1               # скорость: 1, 2, 4, 8 - последняя выбранная
    show_thoughts: bool = True
    show_vision: bool = True     # «что видит»: лучи, интерес, намерение, сводка
    theme: str = "dark"
    world_size: str = "medium"
    volume: int = 60             # 0..100, 0 - звуковое устройство не открывается
    guitar_mode: str = "musician"
    guitar_freq: str = "sometimes"
    guitar_fast: str = "slow"    # что делать со скоростью, пока он играет

    @staticmethod
    def valid(name, value):
        checks = {
            "phrase_freq": lambda v: v in FREQS,
            "speed": lambda v: is_int(v) and v in SPEEDS,
            "show_thoughts": lambda v: isinstance(v, bool),
            "show_vision": lambda v: isinstance(v, bool),
            "theme": lambda v: v in THEMES,
            "world_size": lambda v: v in SIZES,
            "volume": lambda v: is_int(v) and 0 <= v <= 100,
            "guitar_mode": lambda v: v in GUITAR_MODES,
            "guitar_freq": lambda v: v in GUITAR_FREQS,
            "guitar_fast": lambda v: v in GUITAR_FAST,
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
