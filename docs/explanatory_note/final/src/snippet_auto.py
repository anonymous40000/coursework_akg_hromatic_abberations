# auto.py: робастная подгонка поля и карта смещений (фрагмент)
        W = wgt[:, None]
        normal = A.T @ (A * W) + R
        rhs = A.T @ (s * wgt)
        try:
            sol = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return None, s.copy()
        if not np.all(np.isfinite(sol)):
            return None, s.copy()
        res = s - A @ sol
        sig = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.05)
        t = np.abs(res) / (4.685 * sig)
        wgt = prior * np.where(t < 1, (1 - t * t) ** 2, 0.0)
        if (wgt > 0).sum() < 30:
            return None, res
# ...
    gy_n = h // GRID_STEP + 2
    gx_n = w // GRID_STEP + 2
    xs = (np.arange(gx_n) * GRID_STEP).astype(np.float32)
    ys = (np.arange(gy_n) * GRID_STEP).astype(np.float32)
    gx, gy = np.meshgrid(xs, ys)
    px, py = gx.copy(), gy.copy()
    for m in reversed(models):
        dx, dy = _disp(m, px, py)
        px, py = px + dx, py + dy
    return gx, gy, px - gx, py - gy
