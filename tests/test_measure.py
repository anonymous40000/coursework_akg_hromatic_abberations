import numpy as np

from achrom.synthetic import make_dot_frame
from achrom.calibration import measure_displacements


def test_measured_shift_matches_ground_truth():
    frame, center = make_dot_frame(k_red=0.02, k_blue=-0.03)
    r, dr_red, dr_blue = measure_displacements(frame, center)
    assert len(r) == 25
    np.testing.assert_allclose(dr_red, 0.02 * r, atol=0.2)
    np.testing.assert_allclose(dr_blue, -0.03 * r, atol=0.2)


def test_clean_frame_gives_zero_shift():
    frame, center = make_dot_frame(k_red=0.0, k_blue=0.0)
    r, dr_red, dr_blue = measure_displacements(frame, center)
    np.testing.assert_allclose(dr_red, 0.0, atol=0.1)
    np.testing.assert_allclose(dr_blue, 0.0, atol=0.1)
