import time

import cv2
import numpy as np

FG_T = 200
BG_T = 120
REACH_FRAC = 0.10
T0, T1 = 22.0, 65.0
TB0, TB1 = 25.0, 41.0
TR0, TR1 = 12.0, 32.0
EV0, EV1 = 8.0, 28.0
PURP_T = 15.0
TRM0, TRM1 = 15.0, 35.0
K_MIN = 121.0
PS_MAX = 12.0
SPAN_MIN = 30.0
DTHR = 18.0


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
        Lb = 0.299 * R + 0.587 * G + 0.114 * B
        fg = (Lb >= np.percentile(Lb, 90)).astype(np.uint8)
    if fg.sum() < 100 or int(fg.sum()) == fg.size:
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
    fg_density = up(sm(mf))
    valid = (up(sm(mb)) > 1e-3) & (fg_density > 1e-3)
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
    r_far = cv2.remap(R, xs + step * (gx / gn), ys + step * (gy / gn),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ev_r_ray = np.clip((R - r_far - EV0) / (EV1 - EV0), 0.0, 1.0)
    b_low = cv2.resize(cv2.erode(cv2.resize(B, (w // 4, h // 4), interpolation=cv2.INTER_AREA),
                                 _ell(K_MIN / 4)), (w, h), interpolation=cv2.INTER_LINEAR)
    ev_min = np.clip((B - b_low - EV0) / (EV1 - EV0), 0.0, 1.0)
    pocket = (np.clip((90.0 - dist_bg) / 30.0, 0.0, 1.0)
              * np.clip((115.0 - G) / 25.0, 0.0, 1.0)
              * np.clip((100.0 - dist) / 20.0, 0.0, 1.0))
    region = np.clip((cv2.GaussianBlur(((B - G) > 15).astype(np.float32), (0, 0), 14.0) - 0.93) / 0.07, 0.0, 1.0)
    ca_hue = np.clip((-8.0 - (R - G)) / 12.0, 0.0, 1.0)
    gate = np.maximum(1.0 - region, ca_hue)
    evid = np.maximum(ev_ray, ev_min * pocket) * gate

    out = f.copy()
    zone = np.zeros((h, w), np.float32)
    rc, rf_ = ratio(B, mb), ratio(B, mf)
    d_b = B - G * (rc + (rf_ - rc) * alpha)
    rc, rf_ = ratio(R, mb), ratio(R, mf)
    d_r = R - G * (rc + (rf_ - rc) * alpha)
    purple = (d_r > np.maximum(PURP_T, 0.6 * d_b)) | \
        (cv2.GaussianBlur((R > G).astype(np.float32), (0, 0), 2.5) > 0.35)
    for idx, d, t0_, t1_ in ((0, d_b, TB0, TB1), (2, d_r, TR0, TR1)):
        sign = 1.0 if idx == 0 else -1.0
        m = evid * np.clip((sign * d - T0) / (T1 - T0), 0.0, 1.0)
        sharp = np.clip((sign * d - t0_) / (t1_ - t0_), 0.0, 1.0)
        m = np.maximum(m, evid * sharp)
        if idx == 0:
            m = m * (~purple)
        m = m * m * (3 - 2 * m) * gates
        delta = m * d
        if idx == 0:
            delta = np.minimum(delta, np.maximum(B - G + 10.0, 0.0))
        else:
            delta = np.maximum(delta, -np.maximum(G - R + 10.0, 0.0))
        out[..., idx] = f[..., idx] - delta
        zone = np.maximum(zone, m)
    thin_band = np.clip(((B - G) - 0.6 * (R - G) - 5.0) / 15.0, 0.0, 1.0) \
        * (cv2.erode(((R - G) > 15.0).astype(np.uint8)
                     & ((B - G) > 15.0).astype(np.uint8), _ell(11)) == 0)
    evid_mag = np.maximum(np.minimum(ev_r_ray, ev_ray),
                          np.minimum(np.clip((R - cv2.erode(R, _ell(11)) - EV0) / (EV1 - EV0), 0.0, 1.0),
                                     np.clip((B - cv2.erode(B, _ell(11)) - EV0) / (EV1 - EV0), 0.0, 1.0))
                          * np.clip((B - G - 25.0) / 15.0, 0.0, 1.0))
    raw_gates = ((1.0 - region)
                 * np.clip((R - G - 10.0) / 10.0, 0.0, 1.0) * np.clip((B - G - 15.0) / 15.0, 0.0, 1.0))
    sharp_mag = (np.clip((d_b - TB0) / (TB1 - TB0), 0.0, 1.0)
                 * np.clip((d_r - TRM0) / (TRM1 - TRM0), 0.0, 1.0))
    m_mag = evid_mag * raw_gates * sharp_mag
    m_mag = m_mag * m_mag * (3.0 - 2.0 * m_mag)
    m_mag = m_mag * prox * valid * near * (B >= G - 5.0)
    thin_band = thin_band * raw_gates * sharp_mag
    thin_band = thin_band * thin_band * (3.0 - 2.0 * thin_band)
    thin_band = thin_band * prox * valid * (dist > 0) * (dist < reach) * (B >= G - 5.0) \
        * np.clip((fg_density - 0.03) / 0.05, 0.0, 1.0)
    m_mag = np.maximum(m_mag, thin_band)
    delta = np.minimum(m_mag * d_b, np.maximum(out[..., 0] - G + 10.0, 0.0))
    out[..., 0] -= delta
    delta = np.minimum(m_mag * d_r, np.maximum(out[..., 2] - G + 10.0, 0.0))
    out[..., 2] -= delta
    zone = np.maximum(zone, m_mag)
    L = 0.299 * R + 0.587 * G + 0.114 * B
    bright_p = (L >= np.percentile(L, 75)).astype(np.uint8)
    dist_p = cv2.distanceTransform(1 - bright_p, cv2.DIST_L2, 5)
    dps = cv2.GaussianBlur(dist_p, (0, 0), 3.0)
    pgy, pgx = np.gradient(dps)
    pgn = np.hypot(pgx, pgy) + 1e-3
    ps = max(8.0, min(reach / 4.0, PS_MAX))
    ys2, xs2 = np.mgrid[0:h, 0:w].astype(np.float32)
    ddx, ddy = ps * (pgx / pgn), ps * (pgy / pgn)

    def side(ch, sg):
        return cv2.GaussianBlur(cv2.remap(ch, xs2 + sg * ddx, ys2 + sg * ddy,
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), (0, 0), 2.0)

    Gdark, Gbright = side(G, 1.0), side(G, -1.0)
    span = Gbright - Gdark
    alpha_p = np.clip((G - Gdark) / np.where(np.abs(span) < 1e-3, 1e-3, span), 0.0, 1.0)
    band_p = (np.abs(span) > SPAN_MIN) & (alpha_p > 0.02) & (alpha_p < 0.95)
    def off(a, sg, d=4.0):
        return cv2.remap(a, xs2 + sg * d * (pgx / pgn), ys2 + sg * d * (pgy / pgn),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    for idx in (0, 2):
        ch = out[..., idx]
        cd, cb = side(ch, 1.0), side(ch, -1.0)
        exp_p = cd + (cb - cd) * alpha_p
        d_p = ch - exp_p
        cf = f[..., idx]
        cfd, cfb = side(cf, 1.0), side(cf, -1.0)
        df_p = cf - (cfd + (cfb - cfd) * alpha_p)
        signa = np.sign(off(df_p, 1.0)) != np.sign(off(df_p, -1.0))
        m_p = np.clip((np.abs(d_p) - DTHR) / 25.0, 0.0, 1.0) * band_p * signa
        m_p = m_p * m_p * (3.0 - 2.0 * m_p)
        lo = np.minimum(cd, cb) - 8.0
        hi = np.maximum(cd, cb) + 8.0
        delta = m_p * np.clip(d_p, ch - hi, ch - lo)
        out[..., idx] = ch - delta
        zone = np.maximum(zone, m_p)
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
