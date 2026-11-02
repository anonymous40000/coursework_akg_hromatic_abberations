import numpy as np
import cv2

from achrom.synthetic import apply_radial_shift
from achrom.blind import estimate_displacements_blind
from achrom.calibration import fit_radial, radial_model


def make_textured_frame(shape=(400, 400), k_red=0.03, k_blue=-0.02, seed=1):
    
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, shape).astype(np.uint8), (0, 0), 3)
    center = (shape[1] / 2, shape[0] / 2)
    frame = cv2.merge([
        apply_radial_shift(base, center, k_blue),
        base.copy(),
        apply_radial_shift(base, center, k_red),
    ])
    return frame, center


def test_blind_recovers_known_shift():
    frame, center = make_textured_frame()
    r, dr_red, dr_blue = estimate_displacements_blind(frame, center)

    model_red = fit_radial(r, dr_red)
    model_blue = fit_radial(r, dr_blue)
    for rt in [100.0, 150.0, 190.0]:
        assert abs(radial_model(model_red, rt) - 0.03 * rt) < 0.15
        assert abs(radial_model(model_blue, rt) - (-0.02) * rt) < 0.15


def test_clean_texture_gives_zero():
    frame, center = make_textured_frame(k_red=0.0, k_blue=0.0)
    r, dr_red, dr_blue = estimate_displacements_blind(frame, center)
    assert abs(dr_red).max() < 0.15
    assert abs(dr_blue).max() < 0.15
