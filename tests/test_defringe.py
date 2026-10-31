import cv2
import numpy as np

from achrom.defringe import defringe

OLIVE = (35, 60, 46)


def _scene(blue_halo=True):
    h, w = 500, 700
    g = np.full((h, w), 60, np.float32)
    cv2.circle(g, (350, 250), 120, 235, -1)
    g = cv2.GaussianBlur(g, (0, 0), 4)
    r = np.where(g > 0, 46 + (g - 60) * (235 - 46) / 175, 0).astype(np.float32)
    b = 35 + (g - 60) * (240 - 35) / 175
    if blue_halo:
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (350, 250), 150, 1.0, -1)
        halo = cv2.GaussianBlur(m, (0, 0), 14)
        b = np.maximum(b, 35 + 200 * halo * (g < 230))
    f = cv2.merge([b, g, r])
    return np.clip(f, 0, 255).astype(np.uint8)


def _excess(a):
    b, g, r = [c.astype(int) for c in cv2.split(a)]
    return b - np.maximum(r, g)


def test_broad_blue_halo_is_removed():
    f = _scene()
    out, rep = defringe(f, reach_frac=0.2)
    assert (_excess(f) > 40).sum() > 3000
    assert (_excess(out) > 40).sum() < 0.05 * (_excess(f) > 40).sum()


def test_far_from_halo_is_byte_identical():
    f = _scene()
    out, _ = defringe(f, reach_frac=0.2)
    far = np.zeros(f.shape[:2], bool)
    far[:, :40] = far[:, -40:] = far[:40] = far[-40:] = True
    assert np.array_equal(out[far], f[far])


def test_neutral_image_unchanged():
    f = _scene(blue_halo=False)
    g = f[..., 1]
    n = cv2.merge([g, g, g])
    out, _ = defringe(n)
    assert np.array_equal(out, n)


def test_blue_object_away_from_bright_area_unchanged():
    f = np.full((400, 500, 3), OLIVE, np.uint8)
    f[100:300, 150:350] = (230, 60, 40)
    out, _ = defringe(f)
    assert np.array_equal(out, f)


def test_black_frame_next_to_white_unchanged():
    f = _scene(blue_halo=False)
    f[240:260, 100:300] = 0
    out, _ = defringe(f)
    assert np.array_equal(out[240:260, 100:300], f[240:260, 100:300])


def _ca_scene():
    h, w = 500, 700
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xx - 350.0) ** 2 + (yy - 250.0) ** 2
    g = 60 + 175 * np.exp(-d2 / (2 * 30.0 ** 2))
    r = 46 + 189 * np.exp(-d2 / (2 * 18.0 ** 2))
    b = 45 + 190 * np.exp(-d2 / (2 * 60.0 ** 2))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def _cyan(a):
    b, g, r = [c.astype(int) for c in cv2.split(a)]
    return (b - r >= 20) & (b - g <= 15)


def test_cyan_rim_is_removed():
    f = _ca_scene()
    out, _ = defringe(f, reach_frac=0.35)
    band = np.zeros(f.shape[:2], bool)
    band[130:370, 230:470] = True
    fringe = (_excess(f) > 40) & band
    assert fringe.sum() > 2000
    assert (_cyan(out) & fringe).sum() < 0.02 * fringe.sum()


def _gap_scene():
    h, w = 512, 960
    f = np.empty((h, w, 3), np.float32)
    f[..., 1], f[..., 2], f[..., 0] = 65.0, 60.0, 55.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disk, skirt = np.zeros_like(yy), np.zeros_like(yy)
    for cx, cy in ((300.0, 220.0), (620.0, 220.0)):
        d = np.hypot(xx - cx, yy - cy)
        disk = np.maximum(disk, cv2.GaussianBlur((d < 110).astype(np.float32), (0, 0), 3))
        skirt = np.maximum(skirt, np.exp(-0.5 * (np.maximum(d - 110.0, 0) / 60.0) ** 2))
    f[..., 0] += 150.0 * skirt
    f[..., 0] = np.maximum(f[..., 0], 235.0 * disk)
    f[..., 1] = np.maximum(f[..., 1], 235.0 * disk)
    f[..., 2] = np.maximum(f[..., 2], 230.0 * disk)
    wedge = np.zeros((h, w), np.float32)
    tri = np.array([[(460, 240), (410, 512), (510, 512)]], np.int32)
    cv2.fillPoly(wedge, [tri], 1.0)
    for i in range(3):
        f[..., i] = f[..., i] * (1 - wedge) + 50.0 * wedge
    return np.clip(f, 0, 255).astype(np.uint8)


def test_gap_skirt_between_two_bright_plates_is_removed():
    f = _gap_scene()
    gap = np.zeros(f.shape[:2], bool)
    gap[178:270, 420:500] = True
    before = (f[:, :, 0].astype(int) - f[:, :, 1].astype(int) > 20) & gap
    assert before.sum() > 3000
    out, _ = defringe(f, reach_frac=0.25)
    after = (out[:, :, 0].astype(int) - out[:, :, 1].astype(int) > 20) & gap
    assert after.sum() < 0.05 * before.sum()
