import os

import paths


def test_env_variable_replaces_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV, str(tmp_path / "x"))
    assert paths.data_dir() == str(tmp_path / "x")


def test_user_file_creates_parent_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV, str(tmp_path / "deep" / "er"))
    path = paths.user_file("stats.json")
    assert os.path.isdir(os.path.dirname(path))
    assert path == os.path.join(str(tmp_path / "deep" / "er"), "stats.json")


def test_installed_build_keeps_data_in_local_appdata(tmp_path):
    exe = tmp_path / "app" / "AiBoy.exe"
    exe.parent.mkdir()
    got = paths.pick_data_dir(frozen=True, exe=str(exe), env={"LOCALAPPDATA": str(tmp_path / "lad")})
    assert got == os.path.join(str(tmp_path / "lad"), "AiBoy")


def test_portable_build_keeps_data_in_saves_beside_exe(tmp_path):
    exe = tmp_path / "app" / "AiBoy.exe"
    exe.parent.mkdir()
    (exe.parent / "portable.txt").write_text("x", encoding="utf-8")
    got = paths.pick_data_dir(frozen=True, exe=str(exe), env={"LOCALAPPDATA": str(tmp_path / "lad")})
    assert got == os.path.join(str(exe.parent), "saves")


def test_portable_marker_is_ignored_when_running_from_source(tmp_path):
    (tmp_path / "portable.txt").write_text("x", encoding="utf-8")
    got = paths.pick_data_dir(frozen=False, exe=str(tmp_path / "python.exe"),
                              env={"LOCALAPPDATA": str(tmp_path / "lad")})
    assert got == os.path.join(str(tmp_path / "lad"), "AiBoy")


def test_models_are_found_as_resources():
    assert os.path.isdir(paths.resource("models"))
