import time

import cv2
import numpy as np

FG_T = 200
BG_T = 120
REACH_FRAC = 0.10
T0, T1 = 22.0, 65.0
TB0, TB1 = 4.0, 20.0
TR0, TR1 = 8.0, 36.0
EV0, EV1 = 8.0, 28.0
PURP_T = 15.0


def defringe(frame_bgr, reach_frac=REACH_FRAC):
    t0 = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    f = frame_bgr.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    report = {"changed": 0.0, "zone": 0.0}
    fg = ((G >= FG_T) & (R >= G - 25) & (B <= G + 25)).astype(np.uint8)
    if fg.sum() < 100:
        report["time"] = time.perf_counter() - t0
        return frame_bgr.copy(), report
    reach = reach_frac * min(h, w)
    dist = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)
    clean = (G <= BG_T) & (B <= np.maximum(R, G) + 3) & (R <= G + 12) & (G - R <= 45) & (dist > 0)
    dist_bg = cv2.distanceTransform(1 - clean.astype(np.uint8), cv2.DIST_L2, 5)
    near = (dist > 0) & (dist + dist_bg <= reach)
    ds = 4
    sm = lambda a: cv2.GaussianBlur(cv2.resize(a, (w // ds, h // ds), interpolation=cv2.INTER_AREA),
                                    (0, 0), reach / ds)
    up = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    mf, mb = fg.astype(np.float32), clean.astype(np.float32)

    def est(c, m):
        return up(sm(c * m) / np.maximum(sm(m), 1e-4))

    def ratio(c, m):
        return up(np.clip(sm(c * m) / np.maximum(sm(G * m), 5.0), 0.2, 1.6))

    Gf, Gb = est(G, mf), est(G, mb)
    alpha = np.clip((G - Gb) / np.maximum(Gf - Gb, 20.0), 0.0, 1.0)
    valid = (up(sm(mb)) > 1e-3) & (up(sm(mf)) > 1e-3)
    prox = np.clip((reach - dist) / (0.3 * reach), 0.0, 1.0)
    gates = prox * valid * near * (G >= Gb - 15) * (B >= G - 5)

    dist_s = cv2.GaussianBlur(dist, (0, 0), 5.0)
    gy, gx = np.gradient(dist_s)
    gn = np.hypot(gx, gy) + 1e-6
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    step = reach / 3.0
    b_far = cv2.remap(B, xs + step * (gx / gn), ys + step * (gy / gn),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    evid = np.clip((B - b_far - EV0) / (EV1 - EV0), 0.0, 1.0)

    out = f.copy()
    zone = np.zeros((h, w), np.float32)
    rc, rf_ = ratio(B, mb), ratio(B, mf)
    d_b = B - G * (rc + (rf_ - rc) * alpha)
    rc, rf_ = ratio(R, mb), ratio(R, mf)
    d_r = R - G * (rc + (rf_ - rc) * alpha)
    purple = d_r > np.maximum(PURP_T, 0.6 * d_b)
    for idx, d, t0_, t1_ in ((0, d_b, TB0, TB1), (2, d_r, TR0, TR1)):
        sign = 1.0 if idx == 0 else -1.0
        m = evid * np.clip((sign * d - T0) / (T1 - T0), 0.0, 1.0)
        sharp = np.clip((sign * d - t0_) / (t1_ - t0_), 0.0, 1.0)
        m = np.maximum(m, evid * sharp)
        if idx == 0:
            m = m * (~purple)
        m = m * m * (3 - 2 * m) * gates
        out[..., idx] = f[..., idx] - m * d
        zone = np.maximum(zone, m)
    res = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    report["changed"] = float((res != frame_bgr).any(axis=2).mean())
    report["zone"] = float((zone > 0.05).mean())
    report["time"] = time.perf_counter() - t0
    return res, report
