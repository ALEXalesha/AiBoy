import os

import main


def test_selftest_lives_speaks_and_writes_the_result(qapp, tmp_path, monkeypatch):
    out = tmp_path / "selftest.txt"
    home_before = os.environ["AIBOY_HOME"]
    assert main.main(["--selftest", str(out)]) == 0, out.read_text(encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    assert "ИТОГ: работает" in text
    assert "300 шагов жизни" in text and "фраза: «" in text and "звук:" in text
    # данные пользователя не тронуты: самопроверка писала во временную папку
    assert not os.path.exists(os.path.join(home_before, "gallery.json"))
    monkeypatch.setenv("AIBOY_HOME", home_before)
