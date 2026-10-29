import numpy as np
import cv2

from achrom.correction import suppress_glare


def _bgr(b, g, r):
    return np.array([[[b, g, r]]], dtype=np.uint8)


def test_glare_pixel_is_desaturated():
    img = _bgr(240, 170, 90)
    out = suppress_glare(img)
    b, g, r = [int(c) for c in out[0, 0]]
    assert b < 240
    assert abs(b - g) < 40


def test_neutral_pixel_unchanged():
    img = _bgr(241, 243, 240)
    out = suppress_glare(img)
    assert np.array_equal(out, img)


def test_dark_blue_object_unchanged():
    img = _bgr(80, 60, 50)
    out = suppress_glare(img)
    assert np.array_equal(out, img)


def test_uniform_image_unchanged():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert np.array_equal(suppress_glare(img), img)


def test_full_frame_real_glare():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (90, 170, 240)
    img[100:200, 100:200] = (71, 128, 231)
    out = suppress_glare(img)
    b = out[:, :, 0].astype(int)
    g = out[:, :, 1].astype(int)
    assert (b - g).max() < 100


def _halo_scene(halo_b=200, halo_g=120, halo_r=60):
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (halo_b, halo_g, halo_r)
    img[120:180, 120:180] = (255, 255, 255)
    return img


def test_halo_next_to_highlight_is_removed():
    img = _halo_scene()
    out = suppress_glare(img)
    band = out[130:170, 185:200]
    b = band[:, :, 0].astype(int)
    g = band[:, :, 1].astype(int)
    assert (b - g).max() < 45


def test_halo_far_from_highlight_is_kept():
    img = _halo_scene()
    out = suppress_glare(img)
    corner = out[5:40, 5:40]
    b = corner[:, :, 0].astype(int)
    g = corner[:, :, 1].astype(int)
    assert (b - g).min() == 80


def test_halo_becomes_background_colour_not_cyan():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = (40, 110, 60)
    img[140:260, 140:260] = (255, 255, 255)
    ring = np.zeros((400, 400), dtype=np.uint8)
    ring[110:290, 110:290] = 1
    ring[140:260, 140:260] = 0
    img[ring.astype(bool)] = (200, 110, 60)
    out = suppress_glare(img)
    b = int(out[128, 200, 0])
    assert abs(b - 40) <= 25
