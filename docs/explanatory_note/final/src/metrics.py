import numpy as np

from .calibration import measure_displacements


def channel_residual(frame_bgr, center):
    
    r, dr_red, dr_blue = measure_displacements(frame_bgr, center)
    d = np.abs(np.concatenate([dr_red, dr_blue]))
    return d.mean(), d.max()
