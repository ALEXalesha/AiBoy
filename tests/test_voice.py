import io
import wave

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from brain.state import IDX, STATE_DIM
from teacher import voice as teacher_voice
from voice import synth
from voice.net import VoiceNet


def params(**kw):
    p = dict(zip(synth.PARAMS, [0.5] * len(synth.PARAMS)))
    p.update(kw)
    return np.array([p[k] for k in synth.PARAMS])


def crossings(part):
    return int(np.count_nonzero(np.diff(np.signbit(part))))


def state(**kw):
    s = np.zeros(STATE_DIM, np.float32)
    s[IDX["grounded"]] = 1
    for k, v in kw.items():
        s[IDX[k]] = v
    return s


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=len(synth.PARAMS), max_size=len(synth.PARAMS)))
def test_any_parameters_give_a_quiet_enough_sound_of_the_right_length(values):
    p = np.array(values)
    s = synth.synth(p)
    assert len(s) == synth.length_samples(p)
    assert s.dtype == np.float32 and np.isfinite(s).all()
    assert np.abs(s).max() <= 1.0


@pytest.mark.parametrize("length", [0.0, 0.5, 1.0])
def test_length_follows_the_parameter_and_edges_have_no_clicks(length):
    s = synth.synth(params(length=length))
    assert len(s) == round((synth.MIN_LEN + (synth.MAX_LEN - synth.MIN_LEN) * length) * synth.RATE)
    assert 0.15 < np.abs(s).max() <= 0.6
    edge = int(0.001 * synth.RATE)
    assert np.abs(s[:edge]).max() < 0.05 and np.abs(s[-edge:]).max() < 0.05


def test_rising_glide_ends_higher_and_falling_ends_lower():
    for glide, up in ((1.0, True), (0.0, False)):
        s = synth.synth(params(glide=glide, notes=1.0, length=1.0, vibrato=0.0))
        q = len(s) // 4
        assert (crossings(s[-q:]) > crossings(s[:q])) == up


def test_notes_are_on_the_pentatonic_scale():
    for pitch in np.linspace(0, 1, 11):
        for f in synth.note_freqs(params(pitch=pitch, glide=0.9, notes=1.0)):
            semis = 12 * np.log2(f / synth.BASE)
            assert round(semis) % 12 in synth.PENTATONIC
            assert abs(semis - round(semis)) < 1e-9


def test_wav_bytes_are_a_valid_file():
    data = synth.to_wav(synth.synth(params()))
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == synth.RATE


def test_teacher_gives_a_low_falling_sound_for_a_fall_and_a_rising_one_for_curiosity():
    rng = np.random.default_rng(0)
    fall = teacher_voice.target(state(fell=1.0), rng)
    cur = teacher_voice.target(state(curiosity=0.9), rng)
    i = synth.PARAMS.index
    assert fall[i("pitch")] < cur[i("pitch")]
    assert fall[i("glide")] < 0.5 < cur[i("glide")]
    assert pitch_of(synth.synth(fall)) < pitch_of(synth.synth(cur))


def pitch_of(s):
    """Основной тон по автокорреляции середины звука, Гц."""
    mid = s[len(s) // 4: len(s) // 4 + 2048].astype(float)
    ac = np.correlate(mid, mid, mode="full")[len(mid) - 1:]
    lo, hi = synth.RATE // 1000, synth.RATE // 90
    lag = lo + int(np.argmax(ac[lo:hi]))
    return synth.RATE / lag


def test_voice_net_learns_the_teacher():
    rng = np.random.default_rng(1)
    states = rng.random((400, STATE_DIM)).astype(np.float32)
    states[:, IDX["fell"]] = rng.random(400) > 0.7
    targets = np.stack([teacher_voice.target(s, rng, noise=0.0) for s in states])
    net = VoiceNet(seed=0)
    first = net.train_batch(states, targets, lr=0.01)
    for _ in range(800):
        loss = net.train_batch(states, targets, lr=0.01)
    assert loss < 0.3 * first
    out = net.params_for(state(fell=1.0))
    assert out.shape == (len(synth.PARAMS),) and (0 <= out).all() and (out <= 1).all()


def test_voice_net_save_and_load(tmp_path):
    net = VoiceNet(seed=3)
    net.save(tmp_path / "v.npz")
    other = VoiceNet(seed=4).load(tmp_path / "v.npz")
    s = state(curiosity=0.7)
    np.testing.assert_allclose(net.params_for(s), other.params_for(s))
