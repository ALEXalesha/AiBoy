"""Звуковое устройство в памяти для тестов окна: как настоящее, держит буфер (0.25 с, как у
Windows) и проигрывает из него 1/60 с за кадр. Всё проигранное - в played, по кадрам,
поэтому видно, сколько звука вышло после события и не оборвался ли он щелчком."""
import numpy as np

SR = 44100
FRAME = SR // 60


class Device:
    def __init__(self, size=0.25):
        self.size = int(size * SR)
        self.buf = np.zeros(0, np.float32)
        self.played = []                    # массивы по кадрам

    def free(self):
        return (self.size - len(self.buf)) * 2

    def queued(self):
        return len(self.buf)

    def write(self, data):
        x = np.frombuffer(bytes(data), "<i2").astype(np.float32) / 32767
        x = x[:max(0, self.size - len(self.buf))]
        self.buf = np.concatenate([self.buf, x])
        return len(x) * 2

    def frame(self):
        k = min(len(self.buf), FRAME)
        out = np.zeros(FRAME, np.float32)
        out[:k] = self.buf[:k]
        self.buf = self.buf[k:]
        self.played.append(out)
        return out

    def set_volume(self, v):
        pass

    def stop(self):
        self.buf = np.zeros(0, np.float32)

    def stream(self, since=0):
        return np.concatenate(self.played[since:]) if len(self.played) > since else np.zeros(0, np.float32)


def attach(window):
    """Подключить устройство к микшеру окна; насос и проигрывание - вручную, по кадрам."""
    dev = Device()
    window.audio.opener = lambda: dev
    window.audio.set_volume(0)
    window.audio.set_volume(60)
    window.audio.timer.stop()
    return dev


def frame(window, dev):
    window.observe.on_frame(1 / 60)
    window.audio.pump()
    dev.frame()


def tail_after(stream, at):
    """Сколько секунд звука после отсчёта at, пока не стало тихо, и был ли щелчок: скачок
    между соседними отсчётами в хвосте больше, чем у самого звука до события."""
    after = stream[at:]
    loud = np.flatnonzero(np.abs(after) > 1e-3)
    seconds = (loud[-1] + 1) / SR if len(loud) else 0.0
    before = stream[max(0, at - SR // 2):at]
    usual = float(np.max(np.abs(np.diff(before)))) if len(before) > 1 else 0.0
    tail = after[:loud[-1] + 2] if len(loud) else after[:1]
    jump = float(np.max(np.abs(np.diff(np.concatenate([tail, [0.0]]))))) if len(tail) else 0.0
    return seconds, jump, usual
