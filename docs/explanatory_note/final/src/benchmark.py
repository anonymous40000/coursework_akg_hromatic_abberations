import time

import numpy as np

from .correction import correct


def benchmark(coeffs_red, coeffs_blue, sizes, center=None):
    
    rows = []
    for w, h in sizes:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        cx, cy = center or (w / 2, h / 2)
        t0 = time.perf_counter()
        correct(frame, (cx, cy), coeffs_red, coeffs_blue)
        rows.append((w, h, (time.perf_counter() - t0) * 1000))
    return rows
