import cv2
import numpy as np

from .calibration import radial_model


def correct(frame_bgr, center, coeffs_red, coeffs_blue):
    
    b, g, r = cv2.split(frame_bgr)
    return cv2.merge([
        _warp_back(b, center, coeffs_blue),
        g,
        _warp_back(r, center, coeffs_red),
    ])


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
