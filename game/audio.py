"""Микшер игрушек: одно звуковое устройство, 44.1 кГц моно, Int16 (QAudioSink).

Звук вернулся только для игрушек - голоса у человечка нет. Громкость 0 - устройство не
открывается вообще. Нет звуковой карты или QtMultimedia - play() молчит и не падает.
Без экрана (тесты, самопроверка, кадры README) устройство не открывается тоже, но всё
проигранное записывается в history - по нему тесты проверяют, что и в каком порядке
звучало бы.

Бэкенд QtMultimedia - windows: он умеет QAudioSink без ffmpeg, и сборке не нужны пять
библиотек ffmpeg. Звук пишется в устройство по таймеру небольшими порциями и не дальше
LEAD вперёд: так stop() обрывает звук мягко (затухание FADE) и быстро - меньше
чем за LEAD + FADE < 0.1 с. Раньше устройство брало до четверти секунды наперёд, и
остановить уже записанное было нельзя.

Каналы: у каждого своя очередь, в устройство идёт их сумма. «guitar» - звук идущей
попытки, его ведёт окно наблюдения (пауза, меню, новый мир - тишина); остальное -
«toys» (клавиши, «Слушать» в «Попытках»).
"""
import os

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer

SR = 44100
PUMP_MS = 20
LEAD = 0.06              # с звука в устройстве наперёд, не больше
FADE = 0.025             # с - мягкое затухание при остановке и нарастание при продолжении


def _open_default():
    """Открыть устройство по умолчанию. None - нет карты или нет QtMultimedia."""
    os.environ.setdefault("QT_MEDIA_BACKEND", "windows")
    try:
        from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
    except ImportError:
        return None
    device = QMediaDevices.defaultAudioOutput()
    if device.isNull():
        return None
    fmt = QAudioFormat()
    fmt.setSampleRate(SR)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    if not device.isFormatSupported(fmt):
        return None
    return _QtSink(QAudioSink(device, fmt))


class _QtSink:
    def __init__(self, sink):
        self.sink = sink
        self.io = sink.start()

    def free(self):
        return self.sink.bytesFree() if self.io is not None else 0

    def queued(self):
        """Отсчётов, записанных в устройство и ещё не сыгранных."""
        if self.io is None:
            return 0
        return max(0, self.sink.bufferSize() - self.sink.bytesFree()) // 2

    def write(self, data):
        return int(self.io.write(bytes(data))) if self.io is not None else 0

    def set_volume(self, v):
        self.sink.setVolume(float(v))

    def stop(self):
        self.sink.stop()


def _ramp(n, down):
    """Косинус в квадрате: от 1 к 0 (down) или от 0 к 1 - без излома в начале и в конце."""
    x = (np.arange(n, dtype=np.float32) + 0.5) / max(n, 1)
    up = np.sin(0.5 * np.pi * x) ** 2
    return (1.0 - up if down else up).astype(np.float32)


class AudioOut(QObject):
    def __init__(self, volume, opener=None):
        super().__init__()
        offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        self.opener = opener if opener is not None else (None if offscreen else _open_default)
        self.volume = 0
        self.sink = None
        self.channels = {}                # имя -> очередь отсчётов float32
        self.history = []                 # подписи всего, что просили сыграть
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(PUMP_MS)
        self.timer.timeout.connect(self.pump)
        self.set_volume(volume)

    @property
    def pending(self):
        """Всё, что ещё не отдано устройству, смешанное."""
        if not self.channels:
            return np.zeros(0, np.float32)
        n = max(len(q) for q in self.channels.values())
        out = np.zeros(n, np.float32)
        for q in self.channels.values():
            out[:len(q)] += q
        return out

    def set_volume(self, volume):
        self.volume = int(volume)
        if self.volume <= 0:
            self.close()
            return
        if self.sink is None and self.opener is not None:
            try:
                self.sink = self.opener()
            except Exception:  # noqa: BLE001 - звук не должен ронять игру
                self.sink = None
            if self.sink is not None:
                self.timer.start()
        if self.sink is not None:
            self.sink.set_volume(self.volume / 100.0)

    def close(self):
        self.timer.stop()
        if self.sink is not None:
            try:
                self.sink.stop()
            except Exception:  # noqa: BLE001
                pass
        self.sink = None
        self.channels = {}

    def play(self, samples, label=None, channel="toys", fade_in=False):
        """Смешать звук с тем, что уже звучит в канале. False - звука нет (громкость 0, нет
        карты). fade_in - звук начинается с середины (продолжение после паузы): мягкое
        нарастание вместо щелчка."""
        self.history.append(label)
        if self.sink is None:
            return False
        samples = np.array(samples, np.float32)
        if fade_in and len(samples):
            k = min(len(samples), int(FADE * SR))
            samples[:k] *= _ramp(k, down=False)
        q = self.channels.get(channel, np.zeros(0, np.float32))
        if len(samples) > len(q):
            grown = np.zeros(len(samples), np.float32)
            grown[:len(q)] = q
            q = grown
        q[:len(samples)] += samples
        self.channels[channel] = q
        return True

    def playing(self, channel):
        return len(self.channels.get(channel, ())) > 0

    def stop(self, channel=None):
        """Мягко оборвать канал (или все): затухание FADE, остальное выбрасывается."""
        for name in ([channel] if channel is not None else list(self.channels)):
            q = self.channels.pop(name, None)
            if q is None or not len(q):
                continue
            k = min(len(q), int(FADE * SR))
            self.channels[name] = q[:k] * _ramp(k, down=True)

    def stop_all(self):
        self.stop()

    def pump(self):
        if self.sink is None or not self.channels:
            return
        room = self.sink.free() // 2
        if hasattr(self.sink, "queued"):
            room = min(room, int(LEAD * SR) - self.sink.queued())
        longest = max(len(q) for q in self.channels.values())
        n = min(longest, room)
        if n <= 0:
            return
        mix = np.zeros(n, np.float32)
        for q in self.channels.values():
            m = min(n, len(q))
            mix[:m] += q[:m]
        chunk = (np.clip(mix, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        written = max(0, self.sink.write(chunk) // 2)
        for name in list(self.channels):
            rest = self.channels[name][written:]
            if len(rest):
                self.channels[name] = rest
            else:
                del self.channels[name]
