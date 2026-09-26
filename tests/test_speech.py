import re
from pathlib import Path

import numpy as np
import pytest

from brain.state import FIELDS, IDX, STATE_DIM
from speech import vocab
from speech.net import SpeechNet
from teacher import phrases

ROOT = Path(__file__).resolve().parent.parent


def make_state(**kw):
    s = np.zeros(STATE_DIM, np.float32)
    s[IDX["grounded"]] = 1.0
    s[IDX["facing"]] = 1.0
    s[IDX["curiosity"]] = 0.4
    for k, v in kw.items():
        s[IDX[k]] = v
    return s


def entity_state(kind, hue, side=1.0, dist=0.1, sat=0.8, val=0.9):
    return make_state(**{"kind_" + kind: 1.0, "entity": 1.0, "entity_side": side, "entity_dist": dist,
                         "hue_cos": np.cos(2 * np.pi * hue), "hue_sin": np.sin(2 * np.pi * hue),
                         "sat": sat, "val": val})


def test_vocab_roundtrip_keeps_russian_letters_and_drops_the_rest():
    text = "ёжик видит синее дерево!"
    assert vocab.decode(vocab.encode(text)) == text
    assert vocab.decode(vocab.encode("Hello, мир 42")) == ", мир "
    assert vocab.END == 0 and vocab.START == 1 and vocab.SIZE == 2 + len(vocab.CHARS)


def test_colors_are_named_by_hue_and_brightness():
    assert phrases.color_name(0.0, 0.9, 0.9) == "red"
    assert phrases.color_name(0.62, 0.9, 0.9) == "blue"
    assert phrases.color_name(0.33, 0.9, 0.9) == "green"
    assert phrases.color_name(0.5, 0.05, 0.95) == "white"
    assert phrases.color_name(0.5, 0.05, 0.1) == "black"


def test_teacher_agrees_adjectives_with_nouns():
    rng = np.random.default_rng(0)
    for kind in ("tree", "stone", "mushroom"):
        for hue in (0.0, 0.62, 0.33):
            for _ in range(60):
                p = phrases.phrase(entity_state(kind, hue), rng)
                assert not re.search(r"(ый|ий|ой) дерево", p), p
                assert not re.search(r"(ое|ее) (камень|гриб)", p), p
                assert "что-то синий" not in p and "что-то красный" not in p


def test_teacher_speaks_only_words_of_its_dictionary():
    rng = np.random.default_rng(1)
    words = phrases.vocabulary()
    for _ in range(2000):
        s = rng.random(STATE_DIM).astype(np.float32)
        s[IDX["entity_side"]] = rng.choice([-1.0, 1.0])
        s[IDX["want_dir"]] = rng.uniform(-1, 1)
        kinds = [i for i, f in enumerate(FIELDS) if f.startswith("kind_")]
        s[kinds] = 0
        s[rng.choice(kinds)] = 1
        p = phrases.phrase(s, rng)
        assert 0 < len(p) <= vocab.MAX_LEN
        assert vocab.decode(vocab.encode(p)) == p
        assert set(phrases.words(p)) <= words, p


def test_teacher_talks_about_what_matters():
    rng = np.random.default_rng(2)
    fell = [phrases.phrase(make_state(fell=1.0), rng) for _ in range(200)]
    assert sum(p in phrases.FALL for p in fell) > 100
    hill = [phrases.phrase(make_state(hill_right=1.0, flat=0.0), rng) for _ in range(200)]
    assert sum("справа" in p or "повыше" in p for p in hill) > 40
    assert not any("слева" in p for p in hill if "холм" in p)


def test_speech_net_learns_a_small_set_of_phrases():
    rng = np.random.default_rng(0)
    states = np.stack([entity_state("tree", 0.62), make_state(fell=1.0)])
    texts = ["синее дерево!", "ой, больно"]
    net = SpeechNet(hidden=48, seed=0)
    first = None
    for step in range(250):
        loss = net.train_batch(states, texts, lr=0.02)
        first = first if first is not None else loss
    assert loss < 0.1 * first
    assert net.greedy(states[0]) == "синее дерево!"
    assert net.greedy(states[1]) == "ой, больно"
    sample = net.sample(states[0], rng, temperature=0.5)
    assert len(sample) <= vocab.MAX_LEN


def test_speech_net_gradient_matches_numeric():
    net = SpeechNet(hidden=6, seed=3, dtype=np.float64)
    states = np.stack([make_state(fell=1.0), make_state(hill_left=0.8)]).astype(np.float64)
    texts = ["ой", "холм"]
    net.loss_and_grads(states, texts)
    for layer_name, layer in (("cond", net.cond), ("gru", net.gru), ("out", net.out)):
        for name, p, g in layer.params():
            g = g.copy()
            idx = tuple(np.random.default_rng(1).integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + 1e-5
            up = net.loss_and_grads(states, texts)
            p[idx] = old - 1e-5
            down = net.loss_and_grads(states, texts)
            p[idx] = old
            assert g[idx] == pytest.approx((up - down) / 2e-5, rel=1e-4, abs=1e-8), (layer_name, name)


def test_untrained_net_still_writes_a_finite_phrase():
    net = SpeechNet(hidden=32, seed=1)
    text = net.sample(make_state(), np.random.default_rng(0))
    assert len(text) <= vocab.MAX_LEN
    assert set(text) <= set(vocab.CHARS)


def test_speech_net_save_and_load(tmp_path):
    net = SpeechNet(hidden=16, seed=1)
    net.save(tmp_path / "s.npz")
    other = SpeechNet(hidden=16, seed=2)
    other.load(tmp_path / "s.npz")
    s = make_state(fell=1.0)
    assert other.greedy(s) == net.greedy(s)


def test_the_game_never_imports_the_teacher():
    """Главное условие проекта: учитель только при обучении, в игре - одни сети."""
    for folder in ("world", "body", "brain", "speech", "voice", "game", "nn"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"^\s*(from|import)\s+teacher", text, re.M), path
    main = ROOT / "main.py"
    if main.exists():
        assert "teacher" not in main.read_text(encoding="utf-8")
