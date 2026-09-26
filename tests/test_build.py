import os
import re

import pytest

import build
from game.version import VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("world", "body", "speech", "brain")


def nsi():
    with open(os.path.join(ROOT, "installer.nsi"), encoding="utf-8-sig") as f:
        return f.read()


def test_version_is_the_same_in_the_game_and_the_installer():
    assert build.version() == VERSION == "1.0.0"


def test_installer_registers_under_the_app_name_in_russian():
    text = nsi()
    assert re.search(r'!define APP "AiBoy"', text)
    assert 'WriteRegStr HKCU "${REGKEY}" "DisplayName" "${APP}"' in text
    assert '!insertmacro MUI_LANGUAGE "Russian"' in text
    assert 'OutFile "dist\\${APP}-${VERSION}-setup.exe"' in text
    assert "RequestExecutionLevel user" in text


def test_installer_does_not_ship_portable_marker():
    assert 'Delete "$INSTDIR\\portable.txt"' in nsi()


def test_models_are_packed_into_the_build():
    text = open(os.path.join(ROOT, "build.py"), encoding="utf-8").read()
    assert "'models'" in text and "--windowed" in text
    for name in MODELS:
        assert os.path.exists(os.path.join(ROOT, "models", f"{name}.npz")), name
        assert os.path.exists(os.path.join(ROOT, "models", f"{name}.json")), name


def test_icon_is_drawn_in_every_size(tmp_path, qapp):
    Image = pytest.importorskip("PIL.Image")
    path = build.make_icon(str(tmp_path / "icon.ico"))
    with Image.open(path) as ico:
        assert {(16, 16), (32, 32), (256, 256)} <= set(ico.info["sizes"])


def test_trim_removes_only_what_the_app_does_not_need(tmp_path):
    qt = tmp_path / "_internal" / "PySide6"
    (qt / "translations").mkdir(parents=True)
    (qt / "plugins" / "imageformats").mkdir(parents=True)
    keep = [qt / "Qt6Core.dll", qt / "Qt6Gui.dll", qt / "translations" / "qtbase_ru.qm",
            qt / "plugins" / "imageformats" / "qico.dll"]
    (qt / "plugins" / "multimedia").mkdir(parents=True)
    keep += [qt / "Qt6Multimedia.dll", qt / "plugins" / "multimedia" / "windowsmediaplugin.dll"]
    gone = [qt / "opengl32sw.dll", qt / "Qt6Pdf.dll", qt / "translations" / "qtbase_de.qm",
            qt / "avcodec-61.dll", qt / "swscale-8.dll", qt / "plugins" / "multimedia" / "ffmpegmediaplugin.dll",
            qt / "plugins" / "imageformats" / "qpdf.dll"]
    for p in keep + gone:
        p.write_bytes(b"12345")
    assert build.trim(str(tmp_path)) == 5 * len(gone)
    assert all(p.exists() for p in keep)
    assert not any(p.exists() for p in gone)
