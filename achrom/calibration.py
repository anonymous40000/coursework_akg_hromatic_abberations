import cv2
import numpy as np
from math import hypot

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


def _field_basis(u, v, degree):
    cols = [np.ones_like(u), u, v]
    if degree >= 2:
        cols += [u * u, u * v, v * v]
    return np.stack(cols, axis=1)


def fit_field_model(pts, normals, m, shape, degree=1):
    # ponytail: аффинная модель (сдвиг+масштаб+поворот+сдвиг) закрывает
    # латеральную CA и децентрирование; degree=2 (кривизна поля) — если
    # понадобится на мишени с плотными краями
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    scale = max(hypot(w, h) / 2, 1.0)
    u = (pts[:, 0] - cx) / scale
    v = (pts[:, 1] - cy) / scale
    B = _field_basis(u, v, degree)
    nx, ny = normals[:, 0], normals[:, 1]

    def solve(mask):
        rows = np.hstack([B[mask] * nx[mask, None],
                          B[mask] * ny[mask, None]])
        sol, *_ = np.linalg.lstsq(rows, m[mask], rcond=None)
        return sol

    sol = solve(np.ones(len(m), bool))
    pred = (B * sol[: B.shape[1]]).sum(1) * nx + \
           (B * sol[B.shape[1]:]).sum(1) * ny
    inl = np.abs(m - pred) < 1.0
    if inl.sum() >= max(2 * B.shape[1], 10) and inl.sum() < len(m):
        sol = solve(inl)
    return {"sol": sol, "cx": cx, "cy": cy, "scale": scale,
            "degree": degree, "n_basis": B.shape[1]}


def field_displacement(model, xs, ys):
    xs = np.asarray(xs, float)
    shape = xs.shape
    xs = xs.ravel()
    ys = np.asarray(ys, float).ravel()
    u = (xs - model["cx"]) / model["scale"]
    v = (ys - model["cy"]) / model["scale"]
    B = _field_basis(u, v, model["degree"])
    nb = model["n_basis"]
    sol = model["sol"]
    dx = (B * sol[:nb]).sum(1).reshape(shape)
    dy = (B * sol[nb:]).sum(1).reshape(shape)
    return dx, dy


def _nearest(ref, candidates):
    
    d = np.linalg.norm(ref[:, None] - candidates[None], axis=2)
    return candidates[d.argmin(axis=1)]


def _radial_component(ref, shifted, cx, cy, r):
    disp = shifted - ref
    radial = np.stack([ref[:, 0] - cx, ref[:, 1] - cy], axis=1)
    safe_r = np.where(r == 0, 1.0, r)
    proj = (disp * radial).sum(axis=1) / safe_r
    return np.where(r == 0, 0.0, proj)  
