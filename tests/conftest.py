import os
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from hypothesis import settings  # noqa: E402

import offscreen  # noqa: E402

# Без экрана: Qt рисует в память. Ставится до любого импорта Qt.
offscreen.setup()

settings.register_profile("default", max_examples=400, deadline=None)
settings.register_profile("thorough", max_examples=4000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path, monkeypatch):
    """Каждый тест пишет настройки, статистику, галерею и мозг в свою временную папку,
    а не в профиль пользователя."""
    home = tmp_path / "home"
    monkeypatch.setenv("AIBOY_HOME", str(home))
    return home


_app = None


@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон: второе Qt создать не даст."""
    global _app
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    return _app
