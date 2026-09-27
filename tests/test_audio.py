"""Микшер: одно устройство, громкость, ноль - устройство не открывается, нет карты - тишина."""
import numpy as np

from game.audio import AudioOut


class FakeSink:
    def __init__(self):
        self.written = bytearray()
        self.volume = None
        self.stopped = False

    def free(self):
        return 8192

    def write(self, data):
        self.written += data
        return len(data)

    def set_volume(self, v):
        self.volume = v

    def stop(self):
        self.stopped = True


def test_volume_zero_never_opens_the_device(qapp):
    opened = []
    out = AudioOut(0, opener=lambda: opened.append(1) or FakeSink())
    out.play(np.zeros(100, np.float32), "a")
    assert opened == [] and out.sink is None
    assert out.history == ["a"]


def test_volume_opens_and_writes_and_zero_closes(qapp):
    sink = FakeSink()
    out = AudioOut(50, opener=lambda: sink)
    assert out.sink is sink and sink.volume == 0.5
    out.play(np.full(1000, 0.5, np.float32), "note")
    out.pump()
    got = np.frombuffer(bytes(sink.written), "<i2")
    assert len(got) == 1000 and abs(int(got[0]) - 16383) <= 1
    out.set_volume(80)
    assert sink.volume == 0.8
    out.set_volume(0)
    assert sink.stopped and out.sink is None


def test_no_sound_card_is_silent_and_does_not_crash(qapp):
    out = AudioOut(70, opener=lambda: None)
    assert out.sink is None
    assert out.play(np.ones(10, np.float32), "x") is False
    out.pump()
    out.set_volume(30)
    assert out.history == ["x"]


def test_two_sounds_are_mixed_not_queued(qapp):
    sink = FakeSink()
    out = AudioOut(100, opener=lambda: sink)
    out.play(np.full(100, 0.25, np.float32))
    out.play(np.full(50, 0.25, np.float32))
    out.pump()
    got = np.frombuffer(bytes(sink.written), "<i2") / 32767
    assert len(got) == 100
    assert got[:50] == __import__("pytest").approx(0.5, abs=1e-3)
    assert got[50:] == __import__("pytest").approx(0.25, abs=1e-3)


def test_offscreen_default_opens_nothing(qapp):
    out = AudioOut(70)
    assert out.sink is None
    out.play(np.ones(10, np.float32), "silent")
    assert out.history == ["silent"]


def test_device_is_never_fed_more_than_lead_ahead_and_stop_is_soft_and_per_channel(qapp):
    import audio_fake
    from game import audio
    dev = audio_fake.Device()
    out = AudioOut(60, opener=lambda: dev)
    out.timer.stop()
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100).astype(np.float32)
    out.play(tone, "гитара", channel="guitar")
    for _ in range(10):
        out.pump()
        assert dev.queued() <= int(audio.LEAD * audio.SR)
        dev.frame()
    at = sum(len(p) for p in dev.played)
    out.stop("guitar")
    out.play(np.zeros(0, np.float32), "ничего")
    for _ in range(20):
        out.pump()
        dev.frame()
    seconds, jump, usual = audio_fake.tail_after(dev.stream(), at)
    assert 0 < seconds <= audio.LEAD + audio.FADE + 0.001 and jump <= 1.5 * usual
    # другой канал остановка гитары не трогает
    out.play(np.full(1000, 0.1, np.float32), "клавиша")
    out.play(tone, "гитара", channel="guitar")
    out.stop("guitar")
    assert out.playing("toys") and len(out.channels["guitar"]) == int(audio.FADE * audio.SR)
