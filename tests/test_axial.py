import cv2
import numpy as np

from achrom.axial import correct_axial, estimate_sigma
from achrom.synthetic import make_dot_frame, make_texture_frame


def _sharpness(channel):
    return float(cv2.Laplacian(channel.astype(np.float32), cv2.CV_32F).var())


def _defocused_frame(sigma_r, sigma_b):
    gray = make_texture_frame(shape=(192, 192), seed=3)
    red = cv2.GaussianBlur(gray, (0, 0), sigma_r)
    blue = cv2.GaussianBlur(gray, (0, 0), sigma_b)
    return cv2.merge([blue, gray.copy(), red]), gray


def test_estimate_sigma_recovers_known_defocus():
    frame, gray = _defocused_frame(1.2, 0.6)

    assert abs(estimate_sigma(frame[:, :, 2], gray) - 1.2) < 0.2
    assert abs(estimate_sigma(frame[:, :, 0], gray) - 0.6) < 0.2


def test_correct_axial_restores_sharpness_of_blurred_channels():
    frame, gray = _defocused_frame(1.2, 1.0)
    flat = gray.astype(np.float32).ravel()
    before_red = np.corrcoef(frame[:, :, 2].astype(np.float32).ravel(), flat)[0, 1]
    before_blue = np.corrcoef(frame[:, :, 0].astype(np.float32).ravel(), flat)[0, 1]

    fixed, report = correct_axial(frame)

    assert report["changed"] is True
    for index, before in ((2, before_red), (0, before_blue)):
        after = np.corrcoef(fixed[:, :, index].astype(np.float32).ravel(), flat)[0, 1]
        assert after > before
        assert after > 0.9


def test_correct_axial_leaves_sharp_frame_untouched():
    gray = make_texture_frame(shape=(192, 192), seed=5)
    frame = cv2.merge([gray.copy(), gray.copy(), gray.copy()])

    fixed, report = correct_axial(frame)

    assert report["changed"] is False
    assert np.array_equal(fixed, frame)


def test_pipeline_axial_stage_is_opt_in():
    from achrom.pipeline import process_frame

    frame, _ = _defocused_frame(1.5, 1.0)

    _, off = process_frame(frame, enable_axial=False)
    _, on = process_frame(frame, enable_axial=True)

    assert off["axial"] is None
    assert on["axial"]["changed"] is True


def _second_moment(kernel):
    height, width = kernel.shape
    ys, xs = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    return float((((xs - cx) ** 2 + (ys - cy) ** 2) * kernel).sum())


def test_measure_axial_kernels_recovers_known_defocus():
    from achrom.axial import measure_axial_kernels
    from achrom.synthetic import make_dot_frame

    cal, _ = make_dot_frame(shape=(300, 300), sigma_red=1.0, sigma_blue=0.6)

    kernel_b, kernel_r = measure_axial_kernels(cal)

    assert abs(_second_moment(kernel_r) - 2.0) < 0.5
    assert abs(_second_moment(kernel_b) - 0.72) < 0.3


def test_correct_axial_with_measured_kernel_restores_photo():
    from achrom.axial import correct_axial, measure_axial_kernels
    from achrom.synthetic import make_dot_frame

    gray = make_texture_frame(shape=(192, 192), seed=3)
    cal, _ = make_dot_frame(shape=(300, 300), sigma_red=1.0, sigma_blue=0.6)
    kernel_b, kernel_r = measure_axial_kernels(cal)
    red = cv2.GaussianBlur(gray, (0, 0), 1.0)
    blue = cv2.GaussianBlur(gray, (0, 0), 0.6)
    frame = cv2.merge([blue, gray.copy(), red])
    flat = gray.astype(np.float32).ravel()

    fixed, report = correct_axial(frame, kernel_b=kernel_b, kernel_r=kernel_r)

    assert report["method"] == "rl-measured"
    for index in (0, 2):
        before = np.corrcoef(frame[:, :, index].astype(np.float32).ravel(), flat)[0, 1]
        after = np.corrcoef(fixed[:, :, index].astype(np.float32).ravel(), flat)[0, 1]
        assert after > before
        assert after > 0.85


def test_measure_axial_kernels_requires_dot_frame():
    import pytest
    from achrom.axial import measure_axial_kernels

    blank = np.zeros((192, 192), np.uint8)
    frame = cv2.merge([blank, blank, blank])

    with pytest.raises(ValueError):
        measure_axial_kernels(frame)


def test_measure_kernels_survives_subpixel_lateral_shift():
    from achrom.axial import effective_sigma, measure_axial_kernels

    cal, _ = make_dot_frame(
        shape=(300, 300), k_red=0.012, k_blue=0.008,
        sigma_red=1.0, sigma_blue=0.6,
    )
    kernel_b, kernel_r = measure_axial_kernels(cal)

    assert abs(effective_sigma(kernel_r) - 1.0) < 0.12
    assert abs(effective_sigma(kernel_b) - 0.6) < 0.12


def test_estimate_sigma_keeps_native_resolution_on_large_frames():
    from achrom.axial import estimate_sigma

    texture = make_texture_frame((768, 2048), seed=5)
    red = cv2.GaussianBlur(texture, (0, 0), 1.2)
    blue = cv2.GaussianBlur(texture, (0, 0), 0.6)

    assert abs(estimate_sigma(red, texture) - 1.2) < 0.15
    assert abs(estimate_sigma(blue, texture) - 0.6) < 0.15
