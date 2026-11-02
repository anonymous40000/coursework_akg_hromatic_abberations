
import functools
import logging
import os
import subprocess
import threading
import time

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

from achrom.blind import estimate_displacements_blind
from achrom.calibration import fit_radial, radial_model
from achrom.correction import correct

THUMB_W = 560  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("achrom.app")


def log_errors(fn):
    
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            log.info("%s прервано пользователем (Ctrl+C)", fn.__name__)
            return
        except Exception:
            log.exception("Сбой в %s", fn.__name__)
            if not args or not getattr(args[0], "_root_alive", lambda: True)():
                return  
            try:
                messagebox.showerror(
                    "Ошибка",
                    f"Сбой в «{fn.__name__}». Подробности — в app.log",
                )
            except Exception:
                pass
    return wrapper


class App:
    def __init__(self, root):
        self.root = root
        root.title("Коррекция хроматических аберраций")
        self.fixed_frame = None
        self._busy = False
        self._pending = None
        self._last_stem = "result"

        bar = tk.Frame(root)
        bar.pack(fill=tk.X, padx=4, pady=4)
        self._buttons = [
            tk.Button(bar, text="Загрузить снимок", command=self.load_photo)
        ]
        self._buttons[0].pack(side=tk.LEFT, padx=3)

        self.status = tk.Label(
            root, text="Загрузите снимок — коррекция выполнится автоматически.\n"
                       "Результат «до/после» откроется в системном просмотрщике.",
            anchor="w", justify=tk.LEFT,
        )
        self.status.pack(fill=tk.X, padx=6, pady=4)

    

    def _root_alive(self):
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def _run_job(self, job):
        
        if self._busy:
            return
        self._busy = True
        self._pending = None
        for b in self._buttons:
            b.config(state=tk.DISABLED)
        self.status.config(text="Обработка… окно отвечает, подождите пару секунд")

        def worker():
            try:
                self._pending = (True, job())
            except Exception:
                log.exception("Сбой фоновой обработки")
                self._pending = (False, None)

        threading.Thread(target=worker, daemon=True, name="proc").start()
        self.root.after(100, self._poll_pending)

    def _poll_pending(self):
        if self._pending is None:
            self.root.after(100, self._poll_pending)
            return
        ok, res = self._pending
        self._pending = None
        self._busy = False
        for b in self._buttons:
            b.config(state=tk.NORMAL)
        if ok:
            try:
                self._display(*res)
            except Exception:
                log.exception("Сбой при показе результата — "
                              "файлы уже сохранены на диске")
        else:
            messagebox.showerror(
                "Ошибка", "Сбой при обработке. Подробности — в app.log")

    

    @log_errors
    def load_photo(self):
        frame = self._open_image()
        if frame is None:
            return
        stem = self._last_stem
        self._run_job(lambda: self._blind_job(frame, stem))

    

    def _blind_job(self, frame, prefix):
        t_start = time.perf_counter()
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        log.info("Слепая коррекция %dx%d", frame.shape[1], frame.shape[0])
        cur = frame
        acc_r = acc_b = 0.0
        n_edges = 0
        for i in range(3):  
            t0 = time.perf_counter()
            r, dr_red, dr_blue = estimate_displacements_blind(cur, center)
            n_edges = len(r)
            if n_edges == 0:
                raise ValueError("Не найдено достаточно краёв")
            cr = fit_radial(r, dr_red, degree=1)
            cb = fit_radial(r, dr_blue, degree=1)
            rmax = r.max()
            d_r = radial_model(cr, rmax)
            d_b = radial_model(cb, rmax)
            log.info("итерация %d: оценка по %d краям за %.1fс "
                     "(сдвиг R %+.2f, B %+.2f)",
                     i + 1, n_edges, time.perf_counter() - t0, d_r, d_b)
            if max(abs(d_r), abs(d_b)) < 0.5:
                break
            acc_r += d_r
            acc_b += d_b
            t0 = time.perf_counter()
            cur = correct(cur, center, cr, cb)
            log.info("итерация %d: варп за %.1fс", i + 1,
                     time.perf_counter() - t0)
        log.info("Слепая коррекция завершена за %.1fс",
                 time.perf_counter() - t_start)
        unchanged = cur is frame
        status = (
            f"Автокоррекция по {n_edges} краям: сдвиг на краю кадра "
            f"R {acc_r:+.1f}px, B {acc_b:+.1f}px устранён (остаток <0.7px)."
            if not unchanged else "Снимок не требовал коррекции (CA < 0.5px)."
        )
        info = ("Латеральная хроматическая аберрация меньше 0.5px — "
                "снимок не требует коррекции.") if unchanged else None
        diff_path = self._write_outputs(frame, cur, prefix)
        return (frame, cur, diff_path, status, info)

    def _write_outputs(self, before, after, prefix):
        
        t0 = time.perf_counter()
        out = os.path.join("results", prefix)
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(f"{out}_before.png", before)
        cv2.imwrite(f"{out}_after.png", after)
        zb, za = self._corner_zoom(before, after)
        cv2.imwrite(f"{out}_diff.png", np.hstack([zb, za]))
        log.info("PNG-файлы записаны за %.1fс", time.perf_counter() - t0)
        return f"{out}_diff.png"

    

    def _display(self, before, after, diff_path, status, info):
        self.fixed_frame = after
        stem = os.path.basename(diff_path).replace("_diff.png", "")
        out_dir = os.path.dirname(os.path.abspath(diff_path))
        self.status.config(
            text=f"{status}\nРезультаты в папке {out_dir}: "
                 f"{stem}_before.png, {stem}_after.png, {stem}_diff.png")
        
        try:
            subprocess.run(["open", diff_path], check=False)
            log.info("_display: открыт в просмотрщике %s", diff_path)
        except OSError:
            log.warning("_display: не удалось открыть просмотрщик")
        if info:
            messagebox.showinfo("CA не обнаружена", info)
        self.root.lift()

    def _corner_zoom(self, before, after):
        
        h, w = before.shape[:2]
        cx, cy = w / 2, h / 2
        corner = max(
            [(0, 0), (w, 0), (0, h), (w, h)],
            key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
        )
        side = min(h, w) // 3
        x0 = w - side if corner[0] >= cx else 0
        y0 = h - side if corner[1] >= cy else 0
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))

        def zoom(img):
            return cv2.resize(img[sl], (THUMB_W, THUMB_W),
                              interpolation=cv2.INTER_LINEAR)

        return zoom(before), zoom(after)

    

    def _open_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return None
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("Не удалось прочитать изображение: %s", path)
            messagebox.showerror("Ошибка", "Не удалось открыть изображение")
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            self._last_stem = stem or "result"
            log.info("Загружено %s (%dx%d)", path,
                     frame.shape[1], frame.shape[0])
        return frame


if __name__ == "__main__":
    try:
        log.info("Старт приложения")
        root = tk.Tk()
        App(root)
        root.mainloop()
        log.info("Успешный выход")
    except KeyboardInterrupt:
        log.info("Выход по Ctrl+C")
        print("\nУспешный выход")
    except Exception:
        log.exception("Критический сбой приложения")
        raise
