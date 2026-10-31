import time

import cv2
import numpy as np

from .auto import _sample_edges, _measure, _profiles

SCALES = ((4, 40, 48, 18.0), (2, 20, 40, 18.0), (1, 14, 32, 20.0))
MIN_SLOPE = 3.0
BLUR = 2.0
CELL = 12
PER_CELL = 6
DS = 4
MIN_SHIFT = 2.0
MIN_CONTRAST = 60.0
MIN_SUPPORT = 3
FADE_SIGMA = 9.0


def local_register(frame_bgr, min_shift=MIN_SHIFT, scales=SCALES):
    t0 = time.perf_counter()
    cur = frame_bgr
    report = {"edges": 0, "used": {"b": 0, "r": 0}, "peak": {"b": 0.0, "r": 0.0}}
    for scale, s_max, half, sigma in scales:
        cur, rep = _one_pass(cur, min_shift, scale, s_max, half, sigma)
        report["edges"] = max(report["edges"], rep["edges"])
        for k in ("b", "r"):
            report["used"][k] += rep["used"][k]
            report["peak"][k] = max(report["peak"][k], rep["peak"][k])
    report["changed"] = float((cur != frame_bgr).any(axis=2).mean())
    report["time"] = time.perf_counter() - t0
    return cur, report


def _one_pass(frame_bgr, min_shift, scale, s_max, half, sigma):
    H, W = frame_bgr.shape[:2]
    rep = {"edges": 0, "used": {"b": 0, "r": 0}, "peak": {"b": 0.0, "r": 0.0}}
    b, g, r = cv2.split(frame_bgr)
    if scale > 1:
        sm = cv2.resize(frame_bgr, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        sb, sg, sr = cv2.split(sm)
    else:
        sb, sg, sr = b, g, r
    h, w = sg.shape
    edges = _sample_edges(sg, MIN_SLOPE, CELL, PER_CELL, BLUR)
    if edges is None:
        return frame_bgr, rep
    pts, nrm, slope = edges
    rep["edges"] = len(pts)
    k_c = np.array([-0.5 * half, 0.5 * half], np.float32)
    pg = _profiles(sg, pts, nrm, k_c)
    contrast_ok = np.abs(pg[:, 1] - pg[:, 0]) >= MIN_CONTRAST
    out = {"b": b, "r": r}
    for k, sch, full in (("b", sb, b), ("r", sr, r)):
        s, q = _measure(sg, sch, pts, nrm, s_max, half)
        q &= (np.abs(s) < s_max - 2) & contrast_ok
        sig = np.abs(s) * scale >= min_shift
        if (q & sig).sum() < MIN_SUPPORT:
            continue
        dx, dy, fade = _dense_field(pts[q], nrm[q], s[q], slope[q], sig[q], h, w, sigma)
        if scale > 1:
            dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR) * scale
            dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR) * scale
            fade = cv2.resize(fade, (W, H), interpolation=cv2.INTER_LINEAR)
        active = fade > 0.02
        mag = np.hypot(dx, dy)
        if not active.any() or mag[active].max() < min_shift:
            continue
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        warped = cv2.remap(full, xs + dx, ys + dy, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
        alpha = np.clip(fade * 3.0, 0.0, 1.0)
        mixed = full.astype(np.float32) + alpha * (warped.astype(np.float32) - full.astype(np.float32))
        out[k] = np.where(active, np.rint(mixed), full).astype(np.uint8)
        rep["used"][k] = int((q & sig).sum())
        rep["peak"][k] = float(mag.max())
    return cv2.merge([out["b"], g, out["r"]]), rep


def _dense_field(pts, nrm, s, slope, sig, h, w, sigma):
    gh, gw = h // DS + 1, w // DS + 1
    wt = np.clip(slope / np.median(slope), 0.3, 3.0).astype(np.float32)
    ix = np.clip((pts[:, 0] / DS).astype(int), 0, gw - 1)
    iy = np.clip((pts[:, 1] / DS).astype(int), 0, gh - 1)
    acc = np.zeros((3, gh, gw), np.float32)
    np.add.at(acc[0], (iy, ix), wt * s * nrm[:, 0])
    np.add.at(acc[1], (iy, ix), wt * s * nrm[:, 1])
    np.add.at(acc[2], (iy, ix), wt)
    sg = sigma / DS
    acc = np.stack([cv2.GaussianBlur(a, (0, 0), sg) for a in acc])
    wsum = np.maximum(acc[2], 1e-6)
    dxs, dys = acc[0] / wsum, acc[1] / wsum
    cnt = np.zeros((gh, gw), np.float32)
    np.add.at(cnt, (iy[sig], ix[sig]), 1.0)
    sf = FADE_SIGMA / DS
    cnt = cv2.GaussianBlur(cnt, (0, 0), sf) * (2 * np.pi * sf * sf)
    fade = np.clip((cnt - MIN_SUPPORT / 2) / MIN_SUPPORT, 0.0, 1.0)
    fade = fade * fade * (3 - 2 * fade)
    full = lambda a: cv2.resize(a, (gw * DS, gh * DS), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return full(dxs * fade), full(dys * fade), full(fade)
