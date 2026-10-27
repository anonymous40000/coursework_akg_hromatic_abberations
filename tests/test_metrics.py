from achrom.synthetic import make_dot_frame
from achrom.calibration import measure_displacements, fit_radial
from achrom.correction import correct
from achrom.metrics import channel_residual


def test_metric_drops_after_correction():
    calib_frame, center = make_dot_frame(k_red=0.04, k_blue=-0.03)
    r, dr_r, dr_b = measure_displacements(calib_frame, center)
    fixed = correct(
        calib_frame, center, fit_radial(r, dr_r), fit_radial(r, dr_b)
    )

    mean_before, max_before = channel_residual(calib_frame, center)
    mean_after, max_after = channel_residual(fixed, center)

    assert mean_before > 1.0
    assert max_before > mean_before
    assert mean_after < mean_before / 10
    assert max_after < 0.3
