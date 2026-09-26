"""Общее для обучения: потоки BLAS, папка моделей, запись паспорта."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


def limit_threads(n="4"):
    """Процессор делят несколько задач: не больше четырёх потоков BLAS. Звать до numpy."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, n)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def write_passport(name, data):
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, name + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return path


def model_file(name):
    os.makedirs(MODELS, exist_ok=True)
    return os.path.join(MODELS, name)
