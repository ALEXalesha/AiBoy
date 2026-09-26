"""Голоса у человечка нет (решение владельца 26.09.2026). С гитарой (1.1.0) звук вернулся -
но только для игрушек: QtMultimedia трогает один файл, game/audio.py, и только QAudioSink на
бэкенде windows, без ffmpeg."""
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


def code_of(text):
    """Текст без строк документации и комментариев."""
    return re.sub(r'"""[\s\S]*?"""|#.*', "", text)


def test_only_the_mixer_touches_qt_multimedia_and_there_is_no_voice():
    folders = ("nn", "world", "body", "brain", "speech", "game", "music", "train", "teacher", "tools")
    for path in list(ROOT.glob("*.py")) + [p for d in folders for p in (ROOT / d).rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+voice\b", text, re.M), path
        assert "QSoundEffect" not in text, path
        if path.name != "audio.py":
            assert not re.search(r"(from|import)\s+PySide6\.QtMultimedia", code_of(text)), path
    assert not (ROOT / "voice").exists() and not (ROOT / "models" / "voice.npz").exists()


def test_the_game_without_a_screen_does_not_open_audio(tmp_path):
    code = ("import os, sys; sys.path.insert(0, r'%s'); import offscreen; offscreen.setup(force=True)\n"
            "os.environ['AIBOY_HOME'] = r'%s'\n"
            "from PySide6.QtWidgets import QApplication; app = QApplication([])\n"
            "from game.app import MainWindow; w = MainWindow(autostart=False); w.show()\n"
            "w.show_page('observe'); w.observe.advance(120); w.close()\n"
            "print(any(m.startswith('PySide6.QtMultimedia') for m in sys.modules))\n") % (ROOT, tmp_path)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr[-500:]


def test_the_build_takes_the_audio_sink_but_not_ffmpeg():
    assert "PySide6.QtMultimedia" not in build.EXCLUDE
    for name in ("Qt6Multimedia.dll", "windowsmediaplugin.dll"):
        assert not any(name.startswith(p) for p in build.DROP_PREFIXES), name
    for name in ("avcodec-61.dll", "avformat-61.dll", "avutil-59.dll", "swresample-5.dll", "swscale-8.dll",
                 "ffmpegmediaplugin.dll", "Qt6MultimediaWidgets.dll", "Qt6MultimediaQuick.dll"):
        assert any(name.startswith(p) for p in build.DROP_PREFIXES), name


def test_the_brain_has_no_sound_output():
    b = Brain(seed=0)
    assert b.talk.W.shape[1] == 1
    assert "sound" not in Decision.__dataclass_fields__


def test_old_settings_with_sound_fields_load(tmp_path):
    """Файл первой сборки (громкость голоса и поле «sound») читается: громкость снова есть -
    для гитары, лишнее поле забывается."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"volume": 40, "sound": True, "theme": "light", "speed": 2}), encoding="utf-8")
    s = Settings.load(path)
    assert s.theme == "light" and s.speed == 2 and s.volume == 40
    s.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["volume"] == 40 and "sound" not in saved


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
