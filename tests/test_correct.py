import numpy as np

from achrom.synthetic import make_dot_frame
from achrom.calibration import measure_displacements, fit_radial
from achrom.correction import correct


def calibrate(frame, center):
    r, dr_red, dr_blue = measure_displacements(frame, center)
    return fit_radial(r, dr_red), fit_radial(r, dr_blue)


def residual(frame, center):
    r, dr_red, dr_blue = measure_displacements(frame, center)
    return max(abs(dr_red).max(), abs(dr_blue).max())


def test_corrected_frame_channels_align():
    calib_frame, center = make_dot_frame(k_red=0.04, k_blue=-0.03)
    coeffs_red, coeffs_blue = calibrate(calib_frame, center)

    
    test_frame, center2 = make_dot_frame(offset=75, step=50, k_red=0.04, k_blue=-0.03)

    assert residual(test_frame, center2) > 1.0  
    fixed = correct(test_frame, center2, coeffs_red, coeffs_blue)
    assert residual(fixed, center2) < 0.3      

