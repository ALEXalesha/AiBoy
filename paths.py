"""Где лежат ресурсы (модели) и данные (настройки, статистика, галерея, веса мозга).

Данные - в %LOCALAPPDATA%\\AiBoy. Если рядом с собранным exe лежит portable.txt - в папке
saves рядом с ним, на компьютере ничего не остаётся. Переменная AIBOY_HOME подменяет папку
целиком: так тесты, самопроверка и кадры README не трогают данные пользователя.
"""
import os
import sys

APP_NAME = "AiBoy"
ENV = "AIBOY_HOME"


def frozen():
    return getattr(sys, "frozen", False)


def resource_dir():
    if frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource(*parts):
    return os.path.join(resource_dir(), *parts)


def pick_data_dir(frozen, exe, env):
    """Папка данных по признакам запуска - отдельно от sys и os.environ, чтобы проверять."""
    if env.get(ENV):
        return env[ENV]
    if frozen:
        beside = os.path.dirname(os.path.abspath(exe))
        if os.path.exists(os.path.join(beside, "portable.txt")):
            return os.path.join(beside, "saves")
    base = env.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_NAME)


def data_dir():
    return pick_data_dir(frozen(), sys.executable, os.environ)


def user_file(*parts):
    path = os.path.join(data_dir(), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
