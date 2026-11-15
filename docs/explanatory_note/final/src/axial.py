from __future__ import annotations

import time

import cv2
import numpy as np

from .calibration import detect_dots

MAX_SIGMA = 1.2
MIN_SIGMA = 0.2
EST_WIDTH = 512
SIGMA_STEP = 0.1
PATCH_RADIUS = 10
MATCH_RADIUS = 12
MIN_DOTS = 5
BAND_ENERGY = 1e-3
MIN_RATIO = 0.05
SMALL_IMAGE_PIXELS = 2_000_000
RL_ITERATIONS_SMALL = 12
RL_ITERATIONS_LARGE = 6
RL_EPS = 1e-3


def _select_patch(channel, reference):
    height, width = reference.shape
    tile_h, tile_w = min(height, EST_WIDTH), min(width, EST_WIDTH)
    if tile_h == height and tile_w == width:
        return channel.astype(np.float32), reference.astype(np.float32)
    best, best_score = (0, 0), -1.0
    for y in range(0, height - tile_h + 1, EST_WIDTH):
        for x in range(0, width - tile_w + 1, EST_WIDTH):
            score = float(cv2.Laplacian(
                reference[y:y + tile_h, x:x + tile_w], cv2.CV_32F
            ).var())
            if score > best_score:
                best, best_score = (y, x), score
    y, x = best
    return (channel[y:y + tile_h, x:x + tile_w].astype(np.float32),
            reference[y:y + tile_h, x:x + tile_w].astype(np.float32))


def estimate_sigma(channel, reference, max_sigma=MAX_SIGMA):
    if channel.shape != reference.shape:
        raise ValueError("Каналы должны совпадать по размеру")
    channel_small, reference_small = _select_patch(channel, reference)
    channel_small = channel_small - channel_small.mean()
    reference_small = reference_small - reference_small.mean()

    best_error = np.inf
    best_sigma = 0.0
    for sigma in np.arange(0.0, max_sigma + 1e-9, SIGMA_STEP):
        blurred = reference_small if sigma == 0.0 else cv2.GaussianBlur(
            reference_small, (0, 0), sigma
        )
        energy = float(np.sum(blurred * blurred))
        if energy <= 0.0:
            continue
        scale = float(np.sum(channel_small * blurred)) / energy
        error = float(np.sum((channel_small - scale * blurred) ** 2))
        if error < best_error:
            best_error, best_sigma = error, sigma
    return float(best_sigma)


def _second_moment(kernel):
    height, width = kernel.shape
    ys, xs = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    return float((((xs - cx) ** 2 + (ys - cy) ** 2) * kernel).sum())


def effective_sigma(kernel):
    return float(np.sqrt(max(_second_moment(kernel), 0.0) / 2.0))


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
    spectrum = np.fft.rfft2(patch)
    return spectrum * np.exp(2j * np.pi * (fx * (cx - x) + fy * (cy - y)))


def measure_axial_kernels(frame, patch_radius=PATCH_RADIUS):
    _validate_frame(frame)

    reference_centers = detect_dots(frame[:, :, 1])
    if len(reference_centers) < MIN_DOTS:
        raise ValueError("Кадр не содержит точечной мишени")

    size = 2 * patch_radius + 1
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    omega2 = (2.0 * np.pi) ** 2 * (fx ** 2 + fy ** 2)
    height, width = frame.shape[:2]

    kernels = {}
    for index in (0, 2):
        centers = detect_dots(frame[:, :, index])
        cross = None
        power = None
        for gx, gy in reference_centers:
            if len(centers) > 0:
                distances = np.hypot(centers[:, 0] - gx, centers[:, 1] - gy)
                nearest = int(distances.argmin())
                if distances[nearest] > MATCH_RADIUS:
                    continue
                matched = centers[nearest]
            else:
                matched = (gx, gy)
            reference_spectrum = _patch_spectrum(
                frame[:, :, 1], (gx, gy), patch_radius, fx, fy
            )
            channel_spectrum = _patch_spectrum(
                frame[:, :, index], matched, patch_radius, fx, fy
            )
            if reference_spectrum is None or channel_spectrum is None:
                continue
            product = channel_spectrum * np.conj(reference_spectrum)
            cross = product if cross is None else cross + product
            energy = np.abs(reference_spectrum) ** 2
            power = energy if power is None else power + energy
        if cross is None:
            raise ValueError("Кадр не содержит точечной мишени")

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
        sigma = min(sigma, patch_radius / 3.0)
        kernels[index] = _gaussian_kernel(sigma) if sigma > 0.0 else _delta_kernel(
            patch_radius
        )
    return kernels[0], kernels[2]


def _delta_kernel(patch_radius):
    size = 2 * patch_radius + 1
    kernel = np.zeros((size, size), np.float64)
    kernel[patch_radius, patch_radius] = 1.0
    return kernel


def _gaussian_kernel(sigma):
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    taps = cv2.getGaussianKernel(2 * radius + 1, sigma)
    return (taps @ taps.T).astype(np.float64)


def _centered_padded(kernel, shape):
    padded = np.zeros(shape, np.float64)
    height, width = kernel.shape
    padded[:height, :width] = kernel
    return np.roll(padded, (-(height // 2), -(width // 2)), axis=(0, 1))


def _richardson_lucy(channel, kernel, iterations):
    pad = max(kernel.shape) // 2 + 2
    padded = np.pad(channel.astype(np.float32), pad, mode="reflect").astype(np.float64)
    otf = np.fft.rfft2(_centered_padded(kernel, padded.shape))
    adjoint = np.conj(otf)
    estimate = padded.copy()
    for _ in range(iterations):
        blurred = np.fft.irfft2(np.fft.rfft2(estimate) * otf, s=padded.shape)
        ratio = padded / np.maximum(blurred, RL_EPS)
        correction = np.fft.irfft2(np.fft.rfft2(ratio) * adjoint, s=padded.shape)
        estimate *= correction
        np.clip(estimate, 0.0, 255.0, out=estimate)
    result = estimate[pad:-pad, pad:-pad]
    return np.rint(np.clip(result, 0.0, 255.0)).astype(np.uint8)


def _validate_frame(frame):
    if not isinstance(frame, np.ndarray):
        raise TypeError("Ожидается изображение numpy.ndarray")
    if frame.dtype != np.uint8:
        raise TypeError("Изображение должно иметь тип uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Ожидается цветное изображение BGR с тремя каналами")


def correct_axial(frame, max_sigma=MAX_SIGMA, kernel_b=None, kernel_r=None):
    started = time.perf_counter()
    _validate_frame(frame)

    pixels = frame.shape[0] * frame.shape[1]
    iterations = RL_ITERATIONS_SMALL if pixels <= SMALL_IMAGE_PIXELS else RL_ITERATIONS_LARGE
    measured = kernel_b is not None or kernel_r is not None
    reference = frame[:, :, 1]
    output = frame.copy()
    sigmas = {}
    for index, name, kernel in ((0, "b", kernel_b), (2, "r", kernel_r)):
        if kernel is not None:
            sigma = effective_sigma(kernel)
            use_kernel = kernel
        else:
            sigma = estimate_sigma(frame[:, :, index], reference, max_sigma)
            use_kernel = _gaussian_kernel(sigma) if sigma >= MIN_SIGMA else None
        sigmas[name] = sigma
        if sigma >= MIN_SIGMA and use_kernel is not None:
            output[:, :, index] = _richardson_lucy(
                frame[:, :, index], use_kernel, iterations
            )

    report = {
        "changed": bool(np.any(output != frame)),
        "sigma": sigmas,
        "iterations": iterations,
        "method": "rl-measured" if measured else "rl-blind",
        "time": time.perf_counter() - started,
    }
    return output, report
