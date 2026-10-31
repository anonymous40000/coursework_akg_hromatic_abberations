import cv2
import numpy as np

from achrom.auto import correct_auto, residual
from achrom.synthetic import apply_radial_shift


def _scene(shape=(500, 700), seed=3):
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, np.float32)
    for _ in range(140):
        x, y = rng.integers(40, shape[1] - 40), rng.integers(40, shape[0] - 40)
        a, b = rng.integers(10, 50, 2)
        cv2.ellipse(base, (int(x), int(y)), (int(a), int(b)), float(rng.integers(0, 180)),
                    0, 360, float(rng.integers(40, 255)), -1)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    return np.clip(base, 0, 255).astype(np.uint8)


def _aberrated(k_red, k_blue):
    g = _scene()
    center = (g.shape[1] / 2, g.shape[0] / 2)
    frame = cv2.merge([apply_radial_shift(g, center, k_blue), g,
                       apply_radial_shift(g, center, k_red)])
    return frame, g


def _mae(a, b, m=30):
    return float(np.abs(a[m:-m, m:-m].astype(np.float32) - b[m:-m, m:-m].astype(np.float32)).mean())


def test_known_aberration_is_removed():
    frame, g = _aberrated(0.012, -0.018)
    out, rep = correct_auto(frame)
    before = max(_mae(frame[..., 0], g), _mae(frame[..., 2], g))
    after = max(_mae(out[..., 0], g), _mae(out[..., 2], g))
    assert after < 0.25 * before, (before, after)


def test_clean_image_is_untouched():
    g = _scene()
    frame = cv2.merge([g, g, g])
    out, rep = correct_auto(frame)
    assert rep["iterations"] == 0
    assert np.array_equal(out, frame)


def test_green_channel_never_modified():
    frame, _ = _aberrated(0.012, -0.018)
    out, _ = correct_auto(frame)
    assert np.array_equal(out[..., 1], frame[..., 1])


def test_residual_drops_after_correction():
    frame, _ = _aberrated(0.012, -0.018)
    out, _ = correct_auto(frame)
    assert residual(out)["b"] < residual(frame, s_max=24)["b"]
