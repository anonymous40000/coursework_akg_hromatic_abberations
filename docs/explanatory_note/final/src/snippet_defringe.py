# defringe.py: отношения фона/переднего плана и коррекция (фрагмент)
    def est(c, m):
        return up(sm(c * m) / np.maximum(sm(m), 1e-4))

    def ratio(c, m):
        return up(np.clip(sm(c * m) / np.maximum(sm(G * m), 5.0), 0.2, 1.6))
# ...
    Gf, Gb = est(G, mf), est(G, mb)
    alpha = np.clip((G - Gb) / np.maximum(Gf - Gb, 20.0), 0.0, 1.0)
# ...
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
