import cv2
import numpy as np

from achrom.synthetic import make_dot_frame
from achrom.blind import estimate_field_shifts
from achrom.calibration import fit_field_model, field_displacement
from achrom.correction import correct_field


def _affine_frame(shape=(500, 500)):
    frame, center = make_dot_frame(shape=shape, offset=40, step=40)
    h, w = shape
    m_blue = np.float32([[1.006, 0.0, 2.0], [0.0, 1.003, -1.5]])
    b, g, r = cv2.split(frame)
    b = cv2.warpAffine(b, m_blue, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b, g, r]), m_blue


def test_affine_shift_recovered():
    frame, m = _affine_frame()
    pts, normals, m_red, m_blue, ok = estimate_field_shifts(frame)
    assert ok.sum() >= 20
    model = fit_field_model(pts[ok], normals[ok], m_blue[ok], frame.shape)
    h, w = frame.shape[:2]
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    worst = 0.0
    for x, y in [(100, 100), (400, 100), (250, 250), (100, 400), (400, 400)]:
        q = inv @ np.float32([x, y, 1])
        truth = (x - q[0], y - q[1])
        dx, dy = field_displacement(model, np.float32([x]), np.float32([y]))
        worst = max(worst, abs(dx[0] - truth[0]), abs(dy[0] - truth[1]))
    assert worst < 0.3, worst


def test_zero_field_untouched():
    frame, _ = make_dot_frame(shape=(500, 500), offset=40, step=40)
    pts, normals, m_red, m_blue, ok = estimate_field_shifts(frame)
    assert ok.sum() >= 20
    for m in (m_red, m_blue):
        model = fit_field_model(pts[ok], normals[ok], m[ok], frame.shape)
        xs = np.array([100.0, 400.0, 250.0, 100.0, 400.0])
        ys = np.array([100.0, 100.0, 250.0, 400.0, 400.0])
        dx, dy = field_displacement(model, xs, ys)
        assert np.abs(dx).max() < 0.3
        assert np.abs(dy).max() < 0.3


def test_correct_field_aligns_channels():
    frame, _ = _affine_frame()
    pts, normals, m_red, m_blue, ok = estimate_field_shifts(frame)
    mr = fit_field_model(pts[ok], normals[ok], m_red[ok], frame.shape)
    mb = fit_field_model(pts[ok], normals[ok], m_blue[ok], frame.shape)
    out = correct_field(frame, mr, mb)
    pts2, n2, mr2, mb2, ok2 = estimate_field_shifts(out)
    assert np.abs(mb2[ok2]).max() < 0.5
