
import cv2
import numpy as np

from .calibration import fit_radial, radial_model

HALF = 28   
S_MAX = 24  
MIN_R = 20  
NCC_THR = 0.75
MAX_POINTS = 500


def estimate_displacements_blind(frame_bgr, center, degree=1):
    
    b, g, r_ch = [c.astype(np.float32) for c in cv2.split(frame_bgr)]
    h, w = g.shape
    cx, cy = center

    pts = _edge_points(g, cx, cy, h, w)
    if len(pts) == 0:
        return np.array([]), np.array([]), np.array([])

    
    t = np.arange(-HALF, HALF + 1, dtype=np.float32)
    d = pts - np.array([cx, cy], np.float32)
    rad = np.linalg.norm(d, axis=1, keepdims=True)
    u = d / rad
    lines = pts[:, None, :] + t[None, :, None] * u[:, None, :]
    map_x = lines[:, :, 0]
    map_y = lines[:, :, 1]
    pg = cv2.remap(g, map_x, map_y, cv2.INTER_LINEAR)
    pr = cv2.remap(r_ch, map_x, map_y, cv2.INTER_LINEAR)
    pb = cv2.remap(b, map_x, map_y, cv2.INTER_LINEAR)

    r = rad[:, 0]
    dr_red, ok_r = _shift_along_profiles(pg, pr)
    dr_blue, ok_b = _shift_along_profiles(pg, pb)
    keep = ok_r & ok_b
    r, dr_red, dr_blue = r[keep], dr_red[keep], dr_blue[keep]
    if len(r) >= 10:
        inliers = (_inliers(r, dr_red, degree)
                   & _inliers(r, dr_blue, degree))
        if inliers.sum() >= 10:
            r, dr_red, dr_blue = r[inliers], dr_red[inliers], dr_blue[inliers]
    return r, dr_red, dr_blue


def _edge_points(g, cx, cy, h, w):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    ys, xs = np.where(mag > np.percentile(mag, 98))
    m = 2 * HALF + 2
    x = xs.astype(np.float32) - cx
    y = ys.astype(np.float32) - cy
    r = np.hypot(x, y)
    ok = (
        (xs > m) & (xs < w - m) & (ys > m) & (ys < h - m) & (r > MIN_R)
    )
    
    
    gxv, gyv = gx[ys, xs], gy[ys, xs]
    cos_with_radial = np.abs(
        (gxv * x + gyv * y) / (np.hypot(gxv, gyv) * r + 1e-9)
    )
    ok &= cos_with_radial > 0.7
    pts = np.stack([xs[ok], ys[ok]], axis=1).astype(np.float32)
    if len(pts) > MAX_POINTS:
        sel = np.random.default_rng(0).choice(len(pts), MAX_POINTS, False)
        pts = pts[sel]
    return pts


def _shift_along_profiles(pg, pc):
    
    shifts = np.arange(-S_MAX, S_MAX + 1)
    scores = np.empty((pg.shape[0], len(shifts)), np.float32)
    for i, s in enumerate(shifts):
        if s >= 0:
            scores[:, i] = _ncc(pg[:, : pg.shape[1] - s], pc[:, s:])
        else:
            scores[:, i] = _ncc(pg[:, -s:], pc[:, : pc.shape[1] + s])

    peak = scores.argmax(axis=1)
    ok = np.take_along_axis(scores, peak[:, None], 1)[:, 0] > NCC_THR

    dr = shifts[peak].astype(np.float64)
    left = np.take_along_axis(scores, np.clip(peak - 1, 0, len(shifts) - 1)[:, None], 1)[:, 0]
    right = np.take_along_axis(scores, np.clip(peak + 1, 0, len(shifts) - 1)[:, None], 1)[:, 0]
    mid = np.take_along_axis(scores, peak[:, None], 1)[:, 0]
    denom = left - 2 * mid + right
    refine = np.where(denom != 0, 0.5 * (left - right) / np.where(denom != 0, denom, 1.0), 0.0)
    return dr + np.clip(refine, -1, 1), ok


def _ncc(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-9
    return num / den


def _inliers(r, dr, degree):
    
    resid = np.abs(dr - radial_model(fit_radial(r, dr, degree), r))
    return resid < 1.0
