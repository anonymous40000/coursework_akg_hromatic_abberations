import cv2
import numpy as np
import pytest

from achrom.auto import correct_auto, residual
from achrom.synthetic import make_dot_frame


def test_auto_registration_removes_known_lateral_ca():
    source, _ = make_dot_frame(
        shape=(500, 500),
        k_blue=0.035,
        k_red=-0.028,
    )
    before = residual(source)
    corrected, report = correct_auto(source)
    after = residual(corrected)

    assert report["applied"] is True
    assert report["iterations"] >= 1
    assert after["b"] < before["b"] * 0.10
    assert after["r"] < before["r"] * 0.10
    assert after["b_p95"] < before["b_p95"] * 0.20
    assert after["r_p95"] < before["r_p95"] * 0.20


def test_auto_registration_is_identity_without_ca():
    source, _ = make_dot_frame(shape=(500, 500), k_blue=0.0, k_red=0.0)
    corrected, report = correct_auto(source)

    assert np.array_equal(corrected, source)
    assert report["applied"] is False
    assert report["iterations"] == 0
    assert report["converged"] is True


def test_residual_exposes_support_and_tail_metrics():
    source, _ = make_dot_frame(shape=(500, 500), k_blue=0.02, k_red=-0.02)
    result = residual(source)

    for channel in ("b", "r"):
        assert result[channel + "_n"] > 0
        assert 0.0 < result[channel + "_support"] <= 1.0
        assert result[channel + "_p95"] >= result[channel + "_p90"]
        assert result[channel + "_p90"] >= result[channel]


def test_auto_rejects_invalid_input():
    with pytest.raises(ValueError):
        correct_auto(np.zeros((100, 100), dtype=np.uint8))
    with pytest.raises(TypeError):
        correct_auto(np.zeros((100, 100, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        correct_auto(np.zeros((64, 64, 3), dtype=np.uint8))


def test_flat_image_is_not_modified_or_reported_as_converged():
    source = np.full((192, 192, 3), 128, dtype=np.uint8)
    corrected, report = correct_auto(source)

    assert np.array_equal(corrected, source)
    assert report["applied"] is False
    assert report["converged"] is False
    assert report["stop_reason"] == "insufficient_edges"
    assert report["edges"] == 0
