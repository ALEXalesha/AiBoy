"""Микшер игрушек: одно звуковое устройство, 44.1 кГц моно, Int16 (QAudioSink).

Звук вернулся только для игрушек - голоса у человечка нет. Громкость 0 - устройство не
открывается вообще. Нет звуковой карты или QtMultimedia - play() молчит и не падает.
Без экрана (тесты, самопроверка, кадры README) устройство не открывается тоже, но всё
проигранное записывается в history - по нему тесты проверяют, что и в каком порядке
звучало бы.

Бэкенд QtMultimedia - windows: он умеет QAudioSink без ffmpeg, и сборке не нужны пять
библиотек ffmpeg. Звук пишется в устройство по таймеру небольшими порциями: буфер
устройства берёт около четверти секунды за раз.
"""
import os

import numpy as np
from PySide6.QtCore import QObject, QTimer

SR = 44100
PUMP_MS = 20


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

    def write(self, data):
        return int(self.io.write(bytes(data))) if self.io is not None else 0

    def set_volume(self, v):
        self.sink.setVolume(float(v))

    def stop(self):
        self.sink.stop()


class AudioOut(QObject):
    def __init__(self, volume, opener=None):
        super().__init__()
        offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        self.opener = opener if opener is not None else (None if offscreen else _open_default)
        self.volume = 0
        self.sink = None
        self.pending = np.zeros(0, np.float32)
        self.history = []                 # подписи всего, что просили сыграть
        self.timer = QTimer(self)
        self.timer.setInterval(PUMP_MS)
        self.timer.timeout.connect(self.pump)
        self.set_volume(volume)

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
        self.pending = np.zeros(0, np.float32)

    def play(self, samples, label=None):
        """Смешать звук с тем, что уже звучит. False - звука нет (громкость 0, нет карты)."""
        self.history.append(label)
        if self.sink is None:
            return False
        samples = np.asarray(samples, np.float32)
        if len(samples) > len(self.pending):
            grown = np.zeros(len(samples), np.float32)
            grown[:len(self.pending)] = self.pending
            self.pending = grown
        self.pending[:len(samples)] += samples
        return True

    def stop_all(self):
        self.pending = np.zeros(0, np.float32)

    def pump(self):
        if self.sink is None or not len(self.pending):
            return
        n = min(len(self.pending), self.sink.free() // 2)
        if n <= 0:
            return
        chunk = (np.clip(self.pending[:n], -1.0, 1.0) * 32767).astype("<i2").tobytes()
        written = self.sink.write(chunk) // 2
        self.pending = self.pending[max(0, written):]
