import numpy as np
import cv2

from achrom.calibration import detect_dots


def make_frame(dots, shape=(100, 100)):
    
    img = np.full(shape, 255, np.uint8)
    for x, y in dots:
        cv2.circle(img, (x, y), 3, 0, -1)
    return img


def test_detects_all_dots():
    dots = [(20, 30), (60, 70), (80, 15)]
    found = detect_dots(make_frame(dots))
    assert len(found) == 3


def test_positions_within_half_pixel():
    dots = [(20, 30), (60, 70), (80, 15)]
    found = detect_dots(make_frame(dots))
    matched = sorted(map(tuple, found))
    for (fx, fy), (ex, ey) in zip(matched, sorted(dots)):
        assert abs(fx - ex) < 0.5
        assert abs(fy - ey) < 0.5
