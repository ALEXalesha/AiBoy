"""«Мир или гитара» - то же правило, что у любопытства: где интереснее, туда и идёт.

- **Интерес к миру** - медленное среднее ошибки предсказателя на том, что он видит
  (brain.err_ema), в долях её обычного уровня при ходьбе REF_ERR: гуляет по новому месту -
  около 0.5, топчется на знакомом - падает к нулю. Замер: при ходьбе по новым мирам
  err_ema 0.0023-0.0031 (медиана 0.0025), на месте за 30 с падает до 0.0011, за минуту - до
  0.0004.
- **Интерес к музыке** - среднее по последним попыткам новизны его фраз и роста оценки.
- Садится играть, когда музыка интереснее мира на порог (настройка «Как часто берёт
  гитару»), встаёт, когда мир интереснее музыки. Гистерезис: не меньше MIN_ATTEMPTS попыток
  подряд и не меньше MIN_WORLD секунд в мире между сессиями.
- Пока играет, мир для него снова свежеет (он давно его не видел): интерес к миру
  возвращается к FRESH.
"""
import math

REF_ERR = 0.0025
THRESHOLDS = {"rare": 0.3, "sometimes": 0.15, "often": 0.05}
MIN_ATTEMPTS = 3
MIN_WORLD = 20.0
FRESH = 0.55
RECOVER = 0.2
MUSIC_K = 0.35
TAU = 10.0
STOP_MARGIN = 0.05


class Mood:
    def __init__(self, freq="sometimes"):
        self.freq = freq
        self.world = 0.5
        self.music = 0.5
        self.playing = False
        self.attempts = 0
        self.world_time = 0.0

    def world_tick(self, err_ema, dt):
        if self.playing:
            return
        obs = min(max((err_ema or 0.0) / REF_ERR, 0.0), 2.0) / 2.0
        self.world += (obs - self.world) * (1.0 - math.exp(-dt / TAU))
        self.world_time += dt

    def after_attempt(self, novelty, progress):
        """novelty - новизна фразы, progress - рост оценки (0.5 - как обычно), оба 0..1."""
        obs = 0.5 * novelty + 0.5 * progress
        self.music += MUSIC_K * (obs - self.music)
        self.world += RECOVER * (FRESH - self.world)
        self.attempts += 1

    def want_play(self):
        return (not self.playing and self.world_time >= MIN_WORLD
                and self.music - self.world > THRESHOLDS.get(self.freq, THRESHOLDS["sometimes"]))

    def want_stop(self):
        return self.playing and self.attempts >= MIN_ATTEMPTS and self.world - self.music > STOP_MARGIN

    def start(self):
        self.playing = True
        self.attempts = 0

    def stop(self):
        self.playing = False
        self.world_time = 0.0

    def to_dict(self):
        return {"world": self.world, "music": self.music}

    @classmethod
    def from_dict(cls, d, freq):
        m = cls(freq)
        if isinstance(d, dict):
            for k in ("world", "music"):
                v = d.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                    setattr(m, k, float(v))
        return m
