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
                   radius=3):
    
    base = np.full(shape, 255, np.uint8)
    for y in range(offset, shape[0], step):
        for x in range(offset, shape[1], step):
            cv2.circle(base, (x, y), radius, 0, -1)
    center = (shape[1] / 2, shape[0] / 2)
    frame = cv2.merge([
        apply_radial_shift(base, center, k_blue),
        base.copy(),
        apply_radial_shift(base, center, k_red),
    ])  
    return frame, center
