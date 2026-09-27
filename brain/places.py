"""Память мест: что он знает о каждом метре своего мира-кольца.

Для каждого метра - сколько раз он туда приходил, когда был там последний раз и какой
там был интерес: средняя по времени, проведённому там, ошибка предсказателя мозга,
делённая на её скользящее среднее (1 - обычное место, больше - там было неожиданно:
бугры, деревья, кристаллы). Плюс метки - где лежит гитара, где сундук.

Мозг ходьбы помнит только последний шаг; эта память - для навигатора (куда идти) и для
выбора занятий (где что лежит). Она не правило: навигатор - сеть, которая по этой
сводке сама решает, идти ли дальше, повернуть назад или остановиться и осмотреться.
"""
import math

import numpy as np

BIN = 1.0                                      # м - одна клетка памяти
BANDS = ((0.0, 6.0), (6.0, 20.0), (20.0, 50.0))  # м - близко, дальше, далеко
MARK_REACH = 30.0                              # м - на таком расстоянии тяга метки в e раз слабее


class Places:
    def __init__(self, width):
        self.width = float(width)
        self.n = max(1, int(math.ceil(self.width / BIN)))
        self.visits = np.zeros(self.n, np.int32)
        self.last = np.full(self.n, -1e9)
        self.isum = np.zeros(self.n)
        self.icount = np.zeros(self.n, np.int32)
        self.cur = None
        self.marks = {}                        # имя -> x

    def index(self, x):
        return int((x % self.width) / BIN) % self.n

    def visit(self, x, t, interest):
        """Он в точке x в момент t; interest - насколько неожиданно здесь сейчас (1 - как обычно).
        Возвращает (новая ли клетка, сколько раз был в ней до этого)."""
        b = self.index(x)
        entered = b != self.cur
        before = int(self.visits[b])
        if entered:
            self.visits[b] += 1
            self.cur = b
        self.last[b] = t
        if interest is not None and np.isfinite(interest):
            self.isum[b] += float(interest)
            self.icount[b] += 1
        return entered, before

    def interest(self):
        """Интерес по клеткам относительно среднего по знакомым (1 - обычное место); где не
        был - 1: о незнакомом месте он ничего не знает."""
        known = self.icount > 0
        out = np.ones(self.n)
        if known.any():
            raw = self.isum[known] / self.icount[known]
            mean = raw.mean()
            out[known] = raw / (mean + 1e-9)
        return out

    def novelty(self):
        return 1.0 / (1.0 + self.visits)

    def explored(self):
        return float((self.visits > 0).mean())

    def ahead(self, x, facing, d):
        """Клетка на расстоянии d по направлению facing (d < 0 - позади)."""
        return self.index(x + facing * d)

    def band(self, x, facing, lo, hi, novelty=None, interest=None):
        """Средние новизна и интерес клеток от lo до hi метров в сторону facing."""
        novelty = self.novelty() if novelty is None else novelty
        interest = self.interest() if interest is None else interest
        steps = np.arange(lo + BIN / 2, min(hi, self.width / 2), BIN)
        if not len(steps):
            return 0.0, 1.0
        idx = [(self.index(x + facing * d)) for d in steps]
        return float(novelty[idx].mean()), float(interest[idx].mean())

    def distance(self, x, to, facing):
        """Сколько идти от x до to, если идти в сторону facing (по кольцу, от 0 до ширины)."""
        return ((to - x) * facing) % self.width

    def features(self, x, facing, pulls=None):
        """Сводка для навигатора: для «впереди» и «сзади» на трёх расстояниях - новизна и
        интерес (минус 1: обычное место - 0), потом тяга меток впереди и сзади (pulls -
        имя метки -> насколько хочется туда сейчас, 0..1) и доля изученного мира."""
        nov, inter = self.novelty(), self.interest()
        out = []
        for side in (1, -1):
            for lo, hi in BANDS:
                n, i = self.band(x, facing * side, lo, hi, nov, inter)
                out += [n, float(np.clip(i - 1.0, -1.0, 2.0))]
        ahead = behind = 0.0
        for name, want in (pulls or {}).items():
            if name not in self.marks or want <= 0:
                continue
            fwd = self.distance(x, self.marks[name], facing)
            back = self.width - fwd
            if fwd <= back:
                ahead += want * math.exp(-fwd / MARK_REACH)
            else:
                behind += want * math.exp(-back / MARK_REACH)
        out += [ahead, behind, self.explored()]
        return np.array(out, np.float32)


N_FEATURES = 2 * len(BANDS) * 2 + 3
