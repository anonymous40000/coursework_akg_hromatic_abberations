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
K_MIN = 121.0


def _ell(k):
    k = max(3, int(k) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def defringe(frame_bgr, reach_frac=REACH_FRAC):
    t0 = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    f = frame_bgr.astype(np.float32)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    report = {"changed": 0.0, "zone": 0.0, "candidate_pixels": 0, "corrected_pixels": 0,
              "changed_by_channel": {"b": 0, "g": 0, "r": 0},
              "mean_abs_delta": {"b": 0.0, "g": 0.0, "r": 0.0}, "method": "directed-fg-bg-evid"}
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
    ev_ray = np.clip((B - b_far - EV0) / (EV1 - EV0), 0.0, 1.0)
    b_low = cv2.resize(cv2.erode(cv2.resize(B, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
                                 _ell(K_MIN / 4)), (w, h), interpolation=cv2.INTER_LINEAR)
    ev_min = np.clip((B - b_low - EV0) / (EV1 - EV0), 0.0, 1.0)
    pocket = np.clip((80.0 - dist_bg) / 30.0, 0.0, 1.0) * np.clip((100.0 - G) / 25.0, 0.0, 1.0)
    evid = np.maximum(ev_ray, ev_min * pocket)

    out = f.copy()
    zone = np.zeros((h, w), np.float32)
    rc, rf_ = ratio(B, mb), ratio(B, mf)
    d_b = B - G * (rc + (rf_ - rc) * alpha)
    rc, rf_ = ratio(R, mb), ratio(R, mf)
    d_r = R - G * (rc + (rf_ - rc) * alpha)
    purple = (d_r > np.maximum(PURP_T, 0.35 * d_b)) | (R > G)
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
    diff = res != frame_bgr
    per_channel = diff.reshape(-1, 3).sum(axis=0)
    report["changed"] = float(diff.any(axis=2).mean())
    report["zone"] = float((zone > 0.05).mean())
    report["candidate_pixels"] = int((zone > 0.05).sum())
    report["corrected_pixels"] = int(diff.any(axis=2).sum())
    report["changed_by_channel"] = {"b": int(per_channel[0]), "g": int(per_channel[1]), "r": int(per_channel[2])}
    report["mean_abs_delta"] = {
        name: float(np.abs(res[..., i].astype(np.int16) - frame_bgr[..., i].astype(np.int16)).mean())
        for i, name in enumerate(("b", "g", "r"))
    }
    report["method"] = "directed-fg-bg-evid"
    report["time"] = time.perf_counter() - t0
    return res, report
