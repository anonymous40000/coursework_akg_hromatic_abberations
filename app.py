"""Small Tk GUI for automatic microscope chromatic-aberration correction."""

import functools
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from achrom.pipeline import process_frame

RESULTS_DIR = Path("results")
PREVIEW_SIZE = 480

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("achrom.app")


def log_errors(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt:
            log.info("%s прервано пользователем", function.__name__)
            return None
        except Exception as error:
            log.exception("Сбой в %s", function.__name__)
            if args and getattr(args[0], "_root_alive", lambda: False)():
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось выполнить операцию. Подробности записаны в app.log.",
                    parent=args[0].root,
                )
            return None
    return wrapper


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Коррекция хроматических аберраций")
        self.root.minsize(560, 170)
        self._busy = False
        self._closing = False
        self._result_queue = queue.Queue(maxsize=1)
        self._last_stem = "result"
        self._buttons = []

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))

        load_button = tk.Button(
            toolbar,
            text="Загрузить снимок",
            command=self.load_photo,
            width=20,
        )
        load_button.pack(side=tk.LEFT, padx=(0, 8))
        self._buttons.append(load_button)

        self.defringe_var = tk.BooleanVar(value=True)
        cleanup_toggle = tk.Checkbutton(
            toolbar,
            text="Мягкое удаление остаточных цветных ореолов",
            variable=self.defringe_var,
            anchor="w",
        )
        cleanup_toggle.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status = tk.Label(
            root,
            text=(
                "Выберите изображение. Сначала выполняется геометрическое "
                "совмещение каналов, затем — необязательная локальная "
                "коррекция цветных ореолов."
            ),
            anchor="w",
            justify=tk.LEFT,
            wraplength=760,
        )
        self.status.pack(fill=tk.X, padx=10, pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        log.info("GUI инициализирован")

    def _root_alive(self):
        if self._closing:
            return False
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def _on_close(self):
        self._closing = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _run_job(self, job):
        if self._busy or not self._root_alive():
            return
        self._busy = True
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        for button in self._buttons:
            button.config(state=tk.DISABLED)
        self.status.config(text="Обработка… интерфейс остаётся отзывчивым.")

        def worker():
            started = time.perf_counter()
            try:
                result = job()
                self._result_queue.put((True, result))
            except Exception as error:
                log.exception("Ошибка фоновой обработки")
                self._result_queue.put((False, error))
            finally:
                log.info("Фоновая задача завершилась за %.2fс", time.perf_counter() - started)

        threading.Thread(target=worker, daemon=True, name="achrom-worker").start()
        self.root.after(100, self._poll_pending)

    def _poll_pending(self):
        if not self._root_alive():
            return
        try:
            ok, result = self._result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_pending)
            return

        self._busy = False
        for button in self._buttons:
            button.config(state=tk.NORMAL)

        if not ok:
            messagebox.showerror(
                "Ошибка обработки",
                "Обработка не завершилась. Подробности записаны в app.log.",
                parent=self.root,
            )
            self.status.config(text="Обработка завершилась с ошибкой. См. app.log.")
            return

        try:
            self._display(*result)
        except Exception:
            log.exception("Не удалось показать результат; файлы уже сохранены")
            self.status.config(text="Файлы сохранены, но открыть просмотр не удалось.")

    @log_errors
    def load_photo(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        frame = self._read_image(path)
        if frame is None:
            messagebox.showerror(
                "Ошибка",
                "Не удалось прочитать изображение или формат не поддерживается.",
                parent=self.root,
            )
            return

        stem = self._safe_stem(Path(path).stem)
        self._last_stem = stem
        cleanup_enabled = bool(self.defringe_var.get())
        log.info("Загружено %s (%dx%d)", path, frame.shape[1], frame.shape[0])
        self._run_job(lambda: self._process_job(frame, stem, cleanup_enabled))

    @staticmethod
    def _read_image(path):
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            if encoded.size == 0:
                return None
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            log.exception("Не удалось прочитать файл %s", path)
            return None

    @staticmethod
    def _safe_stem(stem):
        value = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
        return value[:100] or "result"

    def _process_job(self, frame, prefix, cleanup_enabled):
        started = time.perf_counter()
        log.info("Начата обработка %dx%d; defringe=%s", frame.shape[1], frame.shape[0], cleanup_enabled)
        final_image, report, stages = process_frame(
            frame,
            enable_defringe=cleanup_enabled,
            return_stages=True,
        )
        geometry = report["geometry"]
        before = report["residual_before_geometry"]
        after = report["residual_after_geometry"]
        fringe = report["defringe"]

        log.info(
            "Геометрия: краёв=%d, итераций=%d, сдвиг peak B=%.3f R=%.3f px, "
            "support B=%.1f%% R=%.1f%%, время=%.2fс",
            geometry["edges"], geometry["iterations"],
            geometry["total_shift"]["b"], geometry["total_shift"]["r"],
            geometry["support"]["b"] * 100, geometry["support"]["r"] * 100,
            geometry["time"],
        )
        log.info(
            "Остаток геометрии B median/p95 %.3f/%.3f px -> %.3f/%.3f px; "
            "R %.3f/%.3f px -> %.3f/%.3f px",
            before["b"], before["b_p95"], after["b"], after["b_p95"],
            before["r"], before["r_p95"], after["r"], after["r_p95"],
        )
        log.info(
            "Дефринж: кандидатов=%d, исправлено пикселей=%d (%.3f%%), "
            "B/G/R=%s, время=%.2fс",
            fringe["candidate_pixels"], fringe["corrected_pixels"],
            fringe["changed"] * 100,
            fringe["changed_by_channel"], fringe["time"],
        )

        paths = self._write_outputs(prefix, stages, report)
        status = self._format_status(report, paths["directory"])
        log.info("Обработка %s завершена за %.2fс", prefix, time.perf_counter() - started)
        return stages["source"], stages["geometry"], final_image, paths, status

    def _write_outputs(self, prefix, stages, report):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        base = RESULTS_DIR / prefix
        paths = {
            "before": base.with_name(base.name + "_before.png"),
            "geometry": base.with_name(base.name + "_geometry.png"),
            "after": base.with_name(base.name + "_after.png"),
            "compare": base.with_name(base.name + "_compare.png"),
            "report": base.with_name(base.name + "_report.json"),
            "directory": str(RESULTS_DIR.resolve()),
        }

        self._write_image(paths["before"], stages["source"])
        self._write_image(paths["geometry"], stages["geometry"])
        self._write_image(paths["after"], stages["final"])
        comparison = self._comparison_image(
            stages["source"], stages["geometry"], stages["final"]
        )
        self._write_image(paths["compare"], comparison)
        with open(paths["report"], "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=True)
        log.info("Результаты сохранены: %s", paths["directory"])
        return paths

    @staticmethod
    def _write_image(path, image):
        ok, encoded = cv2.imencode(path.suffix, image)
        if not ok:
            raise OSError("OpenCV не смог закодировать изображение: %s" % path)
        encoded.tofile(str(path))

    @staticmethod
    def _comparison_image(before, geometry, final):
        h, w = before.shape[:2]
        side = min(h, w, 560)
        difference = cv2.absdiff(before, final).max(axis=2).astype(np.float32)
        score = cv2.GaussianBlur(difference, (0, 0), max(2.0, side / 10.0))
        if float(score.max()) > 0.0:
            _, _, _, (cx, cy) = cv2.minMaxLoc(score)
        else:
            cx, cy = w // 2, h // 2
        x0 = min(max(cx - side // 2, 0), w - side)
        y0 = min(max(cy - side // 2, 0), h - side)
        crop = np.s_[y0:y0 + side, x0:x0 + side]

        def zoom(image):
            return cv2.resize(
                image[crop], (PREVIEW_SIZE, PREVIEW_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )

        before_zoom = zoom(before)
        geometry_zoom = zoom(geometry)
        final_zoom = zoom(final)
        amplified = np.clip(
            cv2.absdiff(before_zoom, final_zoom).astype(np.float32) * 5.0,
            0,
            255,
        ).astype(np.uint8)
        return cv2.hconcat([before_zoom, geometry_zoom, final_zoom, amplified])

    @staticmethod
    def _format_metric(residual, channel, percentile):
        value = residual.get(channel + percentile, float("nan"))
        if not np.isfinite(value):
            return "н/д"
        return "%.2f" % value

    def _format_status(self, report, output_directory):
        geometry = report["geometry"]
        before = report["residual_before_geometry"]
        after = report["residual_after_geometry"]
        fringe = report["defringe"]
        shifts = geometry["total_shift"]
        return (
            "Геометрия: {iterations} ит., максимум B {blue:.2f}px / R {red:.2f}px.\n"
            "Остаток B median/p95: {b0}/{b1} → {b2}/{b3}px; "
            "R median/p95: {r0}/{r1} → {r2}/{r3}px.\n"
            "Надёжные края: B {sb:.0f}%, R {sr:.0f}%. Дефринж: "
            "{changed:.3f}% пикселей; подтверждённых кандидатов {candidates}.\n"
            "Файлы сохранены в {directory}."
        ).format(
            iterations=geometry["iterations"], blue=shifts["b"], red=shifts["r"],
            b0=self._format_metric(before, "b", ""),
            b1=self._format_metric(before, "b", "_p95"),
            b2=self._format_metric(after, "b", ""),
            b3=self._format_metric(after, "b", "_p95"),
            r0=self._format_metric(before, "r", ""),
            r1=self._format_metric(before, "r", "_p95"),
            r2=self._format_metric(after, "r", ""),
            r3=self._format_metric(after, "r", "_p95"),
            sb=geometry["support"]["b"] * 100,
            sr=geometry["support"]["r"] * 100,
            changed=fringe["changed"] * 100,
            candidates=fringe["candidate_pixels"],
            directory=output_directory,
        )

    def _display(self, before, geometry, after, paths, status):
        self.status.config(text=status, wraplength=900)
        log.info("Открытие сравнения %s", paths["compare"])
        self._open_preview(paths["compare"])

    @staticmethod
    def _open_preview(path):
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(
                    ["xdg-open", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except OSError:
            log.exception("Не удалось открыть системный просмотрщик")


if __name__ == "__main__":
    try:
        log.info("Старт приложения")
        root = tk.Tk()
        App(root)
        root.mainloop()
        log.info("Успешный выход")
    except KeyboardInterrupt:
        log.info("Выход по Ctrl+C")
    except Exception:
        log.exception("Критический сбой приложения")
        raise
