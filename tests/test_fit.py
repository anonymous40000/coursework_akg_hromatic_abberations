import numpy as np

from achrom.calibration import radial_model, fit_radial


def test_recovers_known_polynomial():
    coeffs = [0.02, -0.0001, 0.0000005]
    r = np.linspace(10, 200, 30)
    d = sum(c * r**i for i, c in enumerate(coeffs, start=1))
    fitted = fit_radial(r, d)
    for rt in [50.0, 120.0, 190.0]:
        assert abs(radial_model(fitted, rt) - radial_model(coeffs, rt)) < 1e-6


def test_model_is_zero_at_center():
    coeffs = [0.02, -0.0001, 0.0000005]
    assert radial_model(coeffs, 0.0) == 0.0
