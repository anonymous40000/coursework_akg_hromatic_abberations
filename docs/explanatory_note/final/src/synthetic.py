import cv2
import numpy as np


def apply_radial_shift(channel, center, k):
    
    h, w = channel.shape
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    scale = 1.0 / (1.0 + k)
    map_x = cx + (xs - cx) * scale
    map_y = cy + (ys - cy) * scale
    return cv2.remap(channel, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def make_dot_frame(shape=(300, 300), offset=50, step=50, k_red=0.0, k_blue=0.0,
                   radius=3, sigma_red=0.0, sigma_blue=0.0):

    base = np.full(shape, 255, np.uint8)
    for y in range(offset, shape[0], step):
        for x in range(offset, shape[1], step):
            cv2.circle(base, (x, y), radius, 0, -1)
    center = (shape[1] / 2, shape[0] / 2)
    blue = apply_radial_shift(base, center, k_blue)
    red = apply_radial_shift(base, center, k_red)
    if sigma_blue > 0.0:
        blue = cv2.GaussianBlur(blue, (0, 0), sigma_blue)
    if sigma_red > 0.0:
        red = cv2.GaussianBlur(red, (0, 0), sigma_red)
    frame = cv2.merge([blue, base.copy(), red])
    return frame, center


def make_texture_frame(shape=(192, 192), seed=0):
    h, w = shape
    rng = np.random.default_rng(seed)
    img = np.full(shape, 128, np.uint8)
    for _ in range(40):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        radius = int(rng.integers(4, 18))
        cv2.circle(img, (x, y), radius, int(rng.integers(0, 256)), -1)
    for _ in range(25):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, int(rng.integers(0, 256)), int(rng.integers(1, 4)))
    return img
