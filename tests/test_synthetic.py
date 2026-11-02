import numpy as np
import cv2

from achrom.synthetic import apply_radial_shift
from achrom.calibration import detect_dots


def test_dot_moves_radially_by_kr():
    center = (100, 100)
    dot = (150, 100)  
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, dot, 3, 255, -1)

    shifted = apply_radial_shift(img, center, k=0.02)

    (sx, sy), = detect_dots(shifted)
    assert abs(sx - 151.0) < 0.5  
    assert abs(sy - 100.0) < 0.5


def test_center_dot_stays_put():
    center = (100, 100)
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, center, 3, 255, -1)

    (sx, sy), = detect_dots(apply_radial_shift(img, center, k=0.05))
    assert abs(sx - 100.0) < 0.5
    assert abs(sy - 100.0) < 0.5
