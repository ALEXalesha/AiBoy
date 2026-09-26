"""Звуков в AiBoy нет (решение владельца 26.09.2026): ни голоса, ни QtMultimedia, ни в
игре, ни в сборке."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import build
from brain.brain import Brain, Decision
from game.settings import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_no_source_imports_qt_multimedia_or_a_voice():
    for path in list(ROOT.glob("*.py")) + [p for d in ("nn", "world", "body", "brain", "speech", "game",
                                                        "train", "teacher", "tools") for p in (ROOT / d).rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(PySide6\.QtMultimedia|voice\b)", text, re.M), path
        assert "QSoundEffect" not in text, path
    assert not (ROOT / "voice").exists() and not (ROOT / "models" / "voice.npz").exists()


def test_the_running_game_does_not_load_qt_multimedia(tmp_path):
    code = ("import os, sys; sys.path.insert(0, r'%s'); import offscreen; offscreen.setup(force=True)\n"
            "os.environ['AIBOY_HOME'] = r'%s'\n"
            "from PySide6.QtWidgets import QApplication; app = QApplication([])\n"
            "from game.app import MainWindow; w = MainWindow(autostart=False); w.show()\n"
            "w.show_page('observe'); w.observe.advance(120); w.close()\n"
            "print(any(m.startswith('PySide6.QtMultimedia') for m in sys.modules))\n") % (ROOT, tmp_path)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr[-500:]


def test_the_build_leaves_qt_multimedia_and_ffmpeg_out():
    assert "PySide6.QtMultimedia" in build.EXCLUDE
    for name in ("Qt6Multimedia.dll", "avcodec-61.dll", "avformat-61.dll", "avutil-59.dll"):
        assert any(name.startswith(p) for p in build.DROP_PREFIXES), name


def test_the_brain_has_no_sound_output():
    b = Brain(seed=0)
    assert b.talk.W.shape[1] == 1
    assert "sound" not in Decision.__dataclass_fields__


def test_old_settings_with_volume_and_sound_load_and_forget_them(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"volume": 40, "sound": True, "theme": "light", "speed": 2}), encoding="utf-8")
    s = Settings.load(path)
    assert s.theme == "light" and s.speed == 2 and not hasattr(s, "volume")
    s.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "volume" not in saved and "sound" not in saved


def test_old_stats_with_sounds_load(tmp_path):
    from game.stats import Stats
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"worlds": 3, "sounds": 12, "phrases": 5}), encoding="utf-8")
    s = Stats.load(path)
    assert (s.worlds, s.phrases) == (3, 5) and "sounds" not in s.to_dict()


def test_no_sounds_folder_is_created(qapp, _own_data_dir):
    from game.app import MainWindow
    w = MainWindow(autostart=False)
    w.show_page("observe")
    w.observe.advance(300)
    w.close()
    assert not os.path.exists(_own_data_dir / "sounds")
