import cv2
import numpy as np

POLY_DEGREE = 3  


def detect_dots(gray):
    
    if gray.mean() > 127:
        gray = 255 - gray
    _, labels, _, centroids = cv2.connectedComponentsWithStats(
        (gray > 128).astype(np.uint8), connectivity=8
    )
    return centroids[1:]  


def measure_displacements(frame_bgr, center):
    
    b, g, r_ch = cv2.split(frame_bgr)
    pts_g = detect_dots(g)
    pts_r = detect_dots(r_ch)
    pts_b = detect_dots(b)

    cx, cy = center
    matched_r = _nearest(pts_g, pts_r)
    matched_b = _nearest(pts_g, pts_b)

    r = np.hypot(pts_g[:, 0] - cx, pts_g[:, 1] - cy)
    dr_red = _radial_component(pts_g, matched_r, cx, cy, r)
    dr_blue = _radial_component(pts_g, matched_b, cx, cy, r)
    return r, dr_red, dr_blue


def radial_model(coeffs, r):
    
    return sum(c * np.asarray(r) ** i for i, c in enumerate(coeffs, start=1))


def fit_radial(r, d, degree=POLY_DEGREE):
    
    r = np.asarray(r, float)
    A = np.stack([r ** i for i in range(1, degree + 1)], axis=1)
    return np.linalg.lstsq(A, d, rcond=None)[0]


def _nearest(ref, candidates):
    
    d = np.linalg.norm(ref[:, None] - candidates[None], axis=2)
    return candidates[d.argmin(axis=1)]


def _radial_component(ref, shifted, cx, cy, r):
    disp = shifted - ref
    radial = np.stack([ref[:, 0] - cx, ref[:, 1] - cy], axis=1)
    safe_r = np.where(r == 0, 1.0, r)
    proj = (disp * radial).sum(axis=1) / safe_r
    return np.where(r == 0, 0.0, proj)  
