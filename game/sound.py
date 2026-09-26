"""Проигрыватель голоса: сэмплы от синтезатора (voice/synth.py) пишутся в WAV в папке
данных (по кругу в несколько файлов - прошлый звук может ещё звучать) и играются через
QSoundEffect. Нет звуковой карты или QtMultimedia - человечек молчит, но живёт. Без
экрана (тесты, самопроверка, кадры) - файлы пишутся, в колонки ничего не идёт."""
import os
from pathlib import Path

from voice.synth import to_wav

SLOTS = 4


class Player:
    def __init__(self, folder, volume=60):
        self.folder = Path(folder)
        self.volume = volume
        self.slot = 0
        self.played = 0
        self.effects = []
        self.enabled = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.enabled = False
        if not self.enabled:
            return
        try:
            from PySide6.QtMultimedia import QSoundEffect
        except ImportError:
            self.enabled = False
            return
        self.effects = [QSoundEffect() for _ in range(SLOTS)]
        self.set_volume(volume)

    def set_volume(self, volume):
        self.volume = volume
        for e in self.effects:
            e.setVolume(volume / 100)

    def play(self, samples):
        """Записать звук и сыграть. Возвращает путь к файлу (или None, если не записался)."""
        path = self.folder / f"voice{self.slot}.wav"
        try:
            path.write_bytes(to_wav(samples))
        except OSError:
            return None
        self.played += 1
        if self.enabled and self.volume > 0 and self.effects:
            from PySide6.QtCore import QUrl
            effect = self.effects[self.slot]
            effect.stop()
            effect.setSource(QUrl.fromLocalFile(str(path)))
            effect.setVolume(self.volume / 100)
            effect.play()
        self.slot = (self.slot + 1) % SLOTS
        return path
