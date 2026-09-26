"""AiBoy - живой аквариум с нейросетью.

    python main.py                         # окно
    python main.py --selftest итог.txt     # самопроверка без экрана

Самопроверка нужна прежде всего собранному exe: он без консоли, поэтому итог пишется в
файл. Окно создаётся offscreen, данные - во временной папке (настройки, галерея и мозг
пользователя не трогаются): мир, человечек, 300 шагов жизни, фраза, звук.
"""
import argparse
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = 300


def selftest(out_path):
    lines = []
    code = 0
    try:
        import offscreen
        offscreen.setup(force=True)
        os.environ["AIBOY_HOME"] = tempfile.mkdtemp(prefix="aiboy-selftest-")
        import numpy as np
        from PySide6.QtWidgets import QApplication

        import paths
        from game.app import MainWindow
        from game.version import VERSION

        started = time.time()
        app = QApplication.instance() or QApplication([])
        window = MainWindow(autostart=False)
        window.show()
        m = window.models
        lines.append(f"AiBoy {VERSION} самопроверка")
        lines.append(f"собранная версия: {'да' if paths.frozen() else 'нет'}")
        lines.append("обученные сети: " + ", ".join(f"{k} {'да' if v else 'НЕТ'}" for k, v in m.trained.items()))
        if not all(m.trained.values()):
            raise RuntimeError("не все обученные сети нашлись в models")
        obs = window.observe
        obs.new_world(seed=2026)
        life = obs.life
        lines.append(f"мир «{life.world.name}»: {life.world.width:.0f} м, сущностей {len(life.world.entities)}, "
                     f"{'ночь' if life.world.night else 'день'}")
        lines.append(f"человечек {life.body.name}: рост {life.body.height():.2f} м")
        x0 = life.human.px
        obs.advance(STEPS)
        if not np.isfinite([life.human.px, life.human.py]).all():
            raise RuntimeError("человечек улетел в бесконечность")
        lines.append(f"{STEPS} шагов жизни: {life.time:.1f} с, мозг {life.brain.steps} шагов, "
                     f"сдвиг {life.human.px - x0:+.2f} м, путь {life.distance:.2f} м")
        text = life.say()
        obs.chat.add(life.time, text)
        if not text.strip():
            raise RuntimeError("сеть речи не сказала ни буквы")
        lines.append(f"фраза: «{text}»")
        samples, params = life.voice()
        path = window.player.play(samples)
        peak = float(np.abs(samples).max())
        if not (0.05 < peak <= 1.0) or path is None:
            raise RuntimeError("звук не синтезировался")
        lines.append(f"звук: {len(samples) / 22050:.2f} с, пик {peak:.2f}, файл {os.path.basename(str(path))}")
        image = window.grab().toImage()
        if image.isNull() or image.width() < 400:
            raise RuntimeError("окно не нарисовалось")
        lines.append(f"кадр окна {image.width()}x{image.height()}, {time.time() - started:.2f} с")
        window.close()
        app.processEvents()
        lines.append("ИТОГ: работает")
    except Exception:  # noqa: BLE001 - любая ошибка должна попасть в файл
        lines.append("ИТОГ: ошибка")
        lines.append(traceback.format_exc())
        code = 1
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description="AiBoy - живой аквариум с нейросетью")
    ap.add_argument("--selftest", metavar="ФАЙЛ", nargs="?", const="selftest.txt",
                    help="без окна: мир, человечек, 300 шагов жизни, фраза и звук; итог в файл")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args.selftest)
    from game.app import run
    return run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
