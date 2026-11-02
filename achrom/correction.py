import cv2
import numpy as np

from .calibration import radial_model, field_displacement


def correct(frame_bgr, center, coeffs_red, coeffs_blue):

    b, g, r = cv2.split(frame_bgr)
    return cv2.merge([
        _warp_back(b, center, coeffs_blue),
        g,
        _warp_back(r, center, coeffs_red),
    ])


def correct_field(frame_bgr, model_red, model_blue):

    b, g, r = cv2.split(frame_bgr)
    return cv2.merge([
        _warp_back_field(b, model_blue),
        g,
        _warp_back_field(r, model_red),
    ])


def _warp_back_field(channel, model):

    h, w = channel.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = field_displacement(model, xs, ys)
    return cv2.remap(channel, (xs + dx).astype(np.float32),
                     (ys + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


GLARE_BRIGHT = 205.0
GLARE_DOM = 25.0
GLARE_CORE = 235.0
GLARE_REACH = 0.12
GLARE_NEAR_EXCESS = 40.0


def suppress_glare(frame_bgr, bright=GLARE_BRIGHT, dom=GLARE_DOM):
    b, g, r = [c.astype(np.float32) for c in cv2.split(frame_bgr)]
    top = cv2.max(r, g)
    excess = b - top

    darkest = np.minimum(np.minimum(b, g), r)
    core = (darkest >= GLARE_CORE).astype(np.uint8)
    reach = GLARE_REACH * min(frame_bgr.shape[:2])
    dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    near = np.clip((reach - dist) / (0.5 * reach), 0.0, 1.0)

    wb = np.clip((b - bright) / (255.0 - bright), 0.0, 1.0)
    wd = np.clip((excess - dom) / dom, 0.0, 1.0)
    wn = np.clip((excess - GLARE_NEAR_EXCESS) / GLARE_NEAR_EXCESS, 0.0, 1.0)
    far = wb * wd
    halo = near * wn

    clean = (excess < GLARE_NEAR_EXCESS) & (g > 25) & (top < bright - 5)
    k = max(15, int(2.0 * reach)) | 1
    num = cv2.boxFilter(b * clean, -1, (k, k))
    den = cv2.boxFilter(g * clean, -1, (k, k))
    bg = np.clip(num / np.maximum(den, 1.0), 0.2, 1.5) * g
    bg = np.where(den > 1.0, np.minimum(bg, top), top)

    gain = np.maximum(far, halo)
    base = np.where(halo > far, bg, top)
    b = b + gain * (base - b)
    return cv2.merge([b, g, r]).round().astype(np.uint8)


def _warp_back(channel, center, coeffs):
    
    h, w = channel.shape
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    r = np.hypot(dx, dy)

    s = r.copy()
    for _ in range(3):
        s = r + radial_model(coeffs, s)

    scale = np.where(r == 0, 1.0, s / np.where(r == 0, 1.0, r))
    map_x = (cx + dx * scale).astype(np.float32)
    map_y = (cy + dy * scale).astype(np.float32)
    return cv2.remap(channel, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
