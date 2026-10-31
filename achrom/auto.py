import time
from math import hypot

import cv2
import numpy as np

CELL = 48
PER_CELL = 8
HALF = 16
MIN_SLOPE = 14.0
GRID_STEP = 16


def correct_auto(frame_bgr, max_iter=4, stop_px=0.12, orders=1, interp=cv2.INTER_CUBIC,
                 find_center=False):
    h, w = frame_bgr.shape[:2]
    t0 = time.perf_counter()
    b, g, r = cv2.split(frame_bgr)
    chans = {"b": b, "r": r}
    steps = {"b": [], "r": []}
    cur = {"b": b, "r": r}
    centers = {}
    report = {"iterations": 0, "edges": 0, "used": {"b": 0, "r": 0}, "step_rms": []}
    edges = _sample_edges(g)
    if edges is None:
        report["time"] = time.perf_counter() - t0
        return frame_bgr.copy(), report
    pts, nrm, slope = edges
    wt = np.clip(slope / np.median(slope), 0.3, 3.0)
    report["edges"] = len(pts)

    for it in range(max_iter):
        s_max = 24 if it == 0 else 6
        rms_all = 0.0
        new_models = {}
        for k in ("b", "r"):
            s, q = _measure(g, cur[k], pts, nrm, s_max, HALF)
            model = _fit(pts[q], nrm[q], s[q], (h, w), orders, wt=wt[q],
                         center=centers.get(k), search=(find_center and it == 0))
            if model is None:
                continue
            centers[k] = (model["cx"], model["cy"])
            report["used"][k] = int(q.sum())
            rms_all = max(rms_all, _edge_rms(model, pts[q], nrm[q]))
            new_models[k] = model
        report["step_rms"].append(rms_all)
        if rms_all < stop_px:
            break
        for k, model in new_models.items():
            steps[k].append(model)
            cur[k] = _warp_total(chans[k], steps[k], h, w, interp)
        report["iterations"] = it + 1

    report["total_shift"] = {k: (_peak_total(steps[k], h, w) if steps[k] else 0.0)
                             for k in ("b", "r")}
    report["time"] = time.perf_counter() - t0
    report["models"] = steps
    return cv2.merge([cur["b"], g, cur["r"]]), report


def _edge_rms(model, pts, nrm):
    dx, dy = _disp(model, pts[:, 0], pts[:, 1])
    return float(np.percentile(np.hypot(dx, dy), 95))


def residual(frame_bgr, s_max=6, half=16):
    b, g, r = cv2.split(frame_bgr)
    edges = _sample_edges(g)
    if edges is None:
        return {"b": float("nan"), "r": float("nan"), "n": 0}
    pts, nrm, _ = edges
    res = {"n": len(pts)}
    for k, ch in (("b", b), ("r", r)):
        s, q = _measure(g, ch, pts, nrm, s_max, half)
        res[k] = float(np.median(np.abs(s[q]))) if q.sum() >= 10 else float("nan")
    return res


def _sample_edges(g, min_slope=None, cell_size=None, per_cell=None, blur=1.0):
    min_slope = MIN_SLOPE if min_slope is None else min_slope
    CELL_ = CELL if cell_size is None else cell_size
    PER_ = PER_CELL if per_cell is None else per_cell
    h, w = g.shape
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), blur)
    gx = cv2.Scharr(gs, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(gs, cv2.CV_32F, 0, 1) / 32.0
    mag = cv2.magnitude(gx, gy)
    margin = 40
    mask = np.zeros_like(mag, bool)
    mask[margin:h - margin, margin:w - margin] = True
    ys, xs = np.nonzero((mag > min_slope) & mask)
    if len(xs) < 20:
        return None
    nx = gx[ys, xs] / mag[ys, xs]
    ny = gy[ys, xs] / mag[ys, xs]
    xf = xs.astype(np.float32)
    yf = ys.astype(np.float32)
    mp = _sample(mag, xf + nx, yf + ny)
    mm = _sample(mag, xf - nx, yf - ny)
    m0 = mag[ys, xs]
    keep = (m0 >= mp) & (m0 >= mm)
    xs, ys, nx, ny, m0 = xs[keep], ys[keep], nx[keep], ny[keep], m0[keep]
    if len(xs) < 20:
        return None
    ncx = (w + CELL_ - 1) // CELL_
    cell = (ys // CELL_) * ncx + (xs // CELL_)
    order = np.lexsort((-m0, cell))
    cell_s = cell[order]
    first = np.r_[0, np.flatnonzero(np.diff(cell_s)) + 1]
    counts = np.diff(np.r_[first, len(cell_s)])
    rank = np.arange(len(cell_s)) - np.repeat(first, counts)
    sel = order[rank < PER_]
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)
    nrm = np.stack([nx[sel], ny[sel]], axis=1).astype(np.float32)
    return pts, nrm, m0[sel]


def _sample(img, x, y, row=1024):
    n = len(x)
    rows = (n + row - 1) // row
    mx = np.zeros(rows * row, np.float32)
    my = np.zeros(rows * row, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(rows, row), my.reshape(rows, row), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.ravel()[:n]


def _profiles(img, pts, nrm, t, chunk=8000):
    img = img.astype(np.float32)
    out = []
    for i in range(0, len(pts), chunk):
        p, n = pts[i:i + chunk], nrm[i:i + chunk]
        lines = p[:, None, :] + t[None, :, None] * n[:, None, :]
        out.append(cv2.remap(img, lines[:, :, 0].copy(), lines[:, :, 1].copy(),
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
    return np.concatenate(out) if out else np.zeros((0, len(t)), np.float32)


NCC_MIN = 0.75
ENERGY = (0.25, 4.0)


def _measure(g, ch, pts, nrm, s_max, half):
    s, best, energy, inner = _measure_raw(g, ch, pts, nrm, s_max, half)
    ok = inner & (best > NCC_MIN) & (energy > ENERGY[0]) & (energy < ENERGY[1])
    return s, ok


def _measure_raw(g, ch, pts, nrm, s_max, half):
    tg = np.arange(-half, half + 1, dtype=np.float32)
    tc = np.arange(-half - s_max, half + s_max + 1, dtype=np.float32)
    pg = _profiles(g, pts, nrm, tg)
    pc = _profiles(ch, pts, nrm, tc)
    dg = np.gradient(pg, axis=1)
    dc = np.gradient(pc, axis=1)
    win = np.exp(-0.5 * (tg / (0.45 * half)) ** 2).astype(np.float32)
    a = dg * win
    an = np.sqrt((a * a).sum(1)) + 1e-6
    shifts = np.arange(-s_max, s_max + 1)
    sc = np.empty((len(pts), len(shifts)), np.float32)
    for i, s in enumerate(shifts):
        seg = dc[:, s_max + s: s_max + s + len(tg)] * win
        sc[:, i] = (a * seg).sum(1) / (an * (np.sqrt((seg * seg).sum(1)) + 1e-6))
    pk = sc.argmax(1)
    best = np.take_along_axis(sc, pk[:, None], 1)[:, 0]
    inner = (pk > 0) & (pk < len(shifts) - 1)
    l = np.take_along_axis(sc, np.clip(pk - 1, 0, len(shifts) - 1)[:, None], 1)[:, 0]
    rr = np.take_along_axis(sc, np.clip(pk + 1, 0, len(shifts) - 1)[:, None], 1)[:, 0]
    den = l - 2 * best + rr
    frac = np.where(den < -1e-6, 0.5 * (l - rr) / np.where(den < -1e-6, den, -1.0), 0.0)
    s = shifts[pk] + np.clip(frac, -1, 1)
    energy = np.sqrt((dc[:, s_max: s_max + len(tg)] ** 2 * win ** 2).sum(1)) / an
    return s.astype(np.float64), best, energy, inner


def _basis(u, v, orders, rho2_cap=None):
    rho2 = u * u + v * v
    if rho2_cap is not None:
        rho2 = np.minimum(rho2, rho2_cap)
    cols_x = [np.ones_like(u)]
    cols_y = [np.zeros_like(u)]
    cols_x.append(np.zeros_like(u))
    cols_y.append(np.ones_like(u))
    p = np.ones_like(u)
    for _ in range(orders + 1):
        cols_x.append(u * p)
        cols_y.append(v * p)
        p = p * rho2
    return np.stack(cols_x, axis=1), np.stack(cols_y, axis=1)


def _design(pts, nrm, shape, orders, center=None):
    h, w = shape
    scale = hypot(w, h) / 2
    cx, cy = center if center is not None else (w / 2, h / 2)
    u = (pts[:, 0] - cx) / scale
    v = (pts[:, 1] - cy) / scale
    bx, by = _basis(u, v, orders)
    A = (bx * nrm[:, :1] + by * nrm[:, 1:]) * scale
    A[:, :2] /= scale
    return A, scale, np.hypot(u, v)


def _irls(A, s, ridge, prior=None, iters=12):
    n_par = A.shape[1]
    R = np.diag(ridge)
    prior = np.ones(len(s)) if prior is None else prior
    wgt = prior.copy()
    sol = np.zeros(n_par)
    for _ in range(iters):
        W = wgt[:, None]
        sol = np.linalg.solve(A.T @ (A * W) + R, A.T @ (s * wgt))
        res = s - A @ sol
        sig = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.05)
        t = np.abs(res) / (4.685 * sig)
        wgt = prior * np.where(t < 1, (1 - t * t) ** 2, 0.0)
        if (wgt > 0).sum() < 30:
            return None, res
    return sol, s - A @ sol


PRIOR_SD = (0.03, 0.03, 0.02)
NOISE_PX = 0.6


def _ridge(o, n_par):
    ridge = np.zeros(n_par)
    for j in range(o + 1):
        ridge[2 + j] = (NOISE_PX / PRIOR_SD[j]) ** 2
    return ridge


def _fit_at(pts, nrm, s, shape, orders, center, wt=None):
    A, scale, rho = _design(pts, nrm, shape, orders, center)
    A = np.hstack([A, np.ones((len(s), 1))])
    sol, res = _irls(A, s, _ridge(orders, A.shape[1]), wt)
    if sol is None:
        return None, np.inf
    sig = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.05)
    t = np.minimum(np.abs(res) / (4.685 * sig), 1.0)
    cost = float(np.mean(1 - (1 - t * t) ** 3))
    model = {"sol": sol[:-1], "bias": float(sol[-1]), "scale": scale,
             "cx": center[0], "cy": center[1], "orders": orders,
             "rho_max": float(np.quantile(rho, 0.97))}
    return model, cost


def _fit(pts, nrm, s, shape, orders, center=None, search=False, wt=None):
    if len(s) < 60:
        return None
    h, w = shape
    c0 = center if center is not None else (w / 2, h / 2)
    if not search:
        return _fit_at(pts, nrm, s, shape, orders, c0, wt)[0]
    best = (np.inf, None)
    for span, n in ((0.10, 5), (0.025, 3)):
        base = (best[1]["cx"], best[1]["cy"]) if best[1] else c0
        for i in np.linspace(-span, span, n):
            for j in np.linspace(-span, span, n):
                c = (base[0] + i * w, base[1] + j * h)
                m, cost = _fit_at(pts, nrm, s, shape, orders, c, wt)
                if m is not None and cost < best[0]:
                    best = (cost, m)
    return best[1]


def _disp(model, x, y):
    u = (x - model["cx"]) / model["scale"]
    v = (y - model["cy"]) / model["scale"]
    bx, by = _basis(u.ravel(), v.ravel(), model["orders"], model["rho_max"] ** 2)
    sol = model["sol"]
    sc = model["scale"]
    mult = np.ones(len(sol))
    mult[2:] = sc
    dx = (bx * (sol * mult)).sum(1).reshape(u.shape)
    dy = (by * (sol * mult)).sum(1).reshape(u.shape)
    return dx, dy


def _total_map(models, h, w):
    gy_n = h // GRID_STEP + 2
    gx_n = w // GRID_STEP + 2
    xs = np.minimum(np.arange(gx_n) * GRID_STEP, w - 1).astype(np.float32)
    ys = np.minimum(np.arange(gy_n) * GRID_STEP, h - 1).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.copy(), gy.copy()
    for m in reversed(models):
        dx, dy = _disp(m, px, py)
        px, py = px + dx, py + dy
    return gx, gy, px - gx, py - gy


def _warp_total(channel, models, h, w, interp):
    gx, gy, dx, dy = _total_map(models, h, w)
    full_x = _grid_to_full(gx + dx, h, w)
    full_y = _grid_to_full(gy + dy, h, w)
    return cv2.remap(channel, full_x, full_y, interp, borderMode=cv2.BORDER_REPLICATE)


def _grid_to_full(grid, h, w):
    gh, gw = grid.shape
    ys = np.minimum(np.arange(h, dtype=np.float32) / GRID_STEP, gh - 1.001)
    xs = np.minimum(np.arange(w, dtype=np.float32) / GRID_STEP, gw - 1.001)
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(grid.astype(np.float32), mx, my, cv2.INTER_LINEAR)


def _peak_shift(model, h, w):
    gx, gy, dx, dy = _total_map([model], h, w)
    return float(np.hypot(dx, dy).max())


def _peak_total(models, h, w):
    gx, gy, dx, dy = _total_map(models, h, w)
    return float(np.hypot(dx, dy).max())
