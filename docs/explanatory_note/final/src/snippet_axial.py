# axial.py: спектральная оценка ядра расфокусировки (фрагмент)
def _patch_spectrum(channel, center, patch_radius, fx, fy):
    height, width = channel.shape
    cx, cy = center
    x, y = int(round(cx)), int(round(cy))
    if x - patch_radius < 0 or y - patch_radius < 0:
        return None
    if x + patch_radius >= width or y + patch_radius >= height:
        return None
    patch = 255.0 - channel[
        y - patch_radius:y + patch_radius + 1,
        x - patch_radius:x + patch_radius + 1,
    ].astype(np.float64)
    patch = patch - np.median(patch)
    patch[patch < 0.0] = 0.0
# ...
        ratio = np.clip(np.real(cross) / np.maximum(power, 1e-12), 0.0, 1.0)
        band = (power > BAND_ENERGY * float(power.max())) & (ratio > MIN_RATIO)
        band &= omega2 > 0.0
        sigma = 0.0
        if np.any(band):
            weights = power[band]
            logs = np.log(ratio[band])
            frequencies = omega2[band]
            sigma_sq = -2.0 * float((weights * frequencies * logs).sum()) / float(
                (weights * frequencies * frequencies).sum()
            )
            sigma = float(np.sqrt(max(sigma_sq, 0.0)))
