import cv2
import numpy as np

from achrom.local import local_register


def _halo_frame(shift=12, blur=5.0):
    h, w = 600, 800
    g = np.full((h, w), 40, np.uint8)
    cv2.ellipse(g, (400, 300), (170, 120), 0, 0, 360, 235, -1)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (400, 300), (170 + shift, 120 + shift), 0, 0, 360, 1.0, -1)
    b = cv2.GaussianBlur(40 + 195 * mask, (0, 0), blur).astype(np.uint8)
    return cv2.merge([b, g, g]), g


def test_halo_around_bright_object_is_pulled_back():
    frame, g = _halo_frame()
    out, rep = local_register(frame)
    ring = np.zeros(g.shape, bool)
    cv2.ellipse(ring.view(np.uint8), (400, 300), (190, 140), 0, 0, 360, 1, 20)
    before = np.abs(frame[..., 0].astype(int) - g.astype(int))[ring].mean()
    after = np.abs(out[..., 0].astype(int) - g.astype(int))[ring].mean()
    assert after < 0.4 * before, (before, after)


def test_far_from_edges_is_byte_identical():
    frame, _ = _halo_frame()
    out, _ = local_register(frame)
    far = np.zeros(frame.shape[:2], bool)
    far[:, :60] = far[:, -60:] = far[:60] = far[-60:] = True
    assert np.array_equal(out[far], frame[far])


def test_neutral_image_unchanged():
    _, g = _halo_frame()
    frame = cv2.merge([g, g, g])
    out, rep = local_register(frame)
    assert np.array_equal(out, frame)


def test_pure_colour_edge_unchanged():
    frame = np.full((400, 500, 3), 40, np.uint8)
    frame[:, 250:, 0] = 230
    out, _ = local_register(frame)
    assert np.array_equal(out, frame)
