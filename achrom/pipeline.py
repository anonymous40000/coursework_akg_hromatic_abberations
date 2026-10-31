"""Public processing pipeline for microscope RGB images."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .auto import correct_auto, residual
from .defringe import defringe


def process_frame(
    frame_bgr: np.ndarray,
    enable_defringe: bool = True,
    return_stages: bool = False,
):
    """Run geometric lateral-CA correction followed by conservative cleanup.

    Geometry and colour cleanup are measured and reported separately.  The
    residual-shift metric is intentionally computed before and after geometry,
    not after defringe, because a colour edit must never masquerade as geometric
    channel registration.

    Args:
        frame_bgr: 8-bit BGR source image.
        enable_defringe: Enable conservative local colour-fringe suppression.
        return_stages: Also return the image after geometric registration.

    Returns:
        (final_image, report), or (final_image, report, stages) when requested.
    """
    started = time.perf_counter()
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("Ожидается изображение numpy.ndarray")
    if frame_bgr.dtype != np.uint8:
        raise TypeError("Изображение должно иметь тип uint8")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Ожидается цветное изображение BGR с тремя каналами")

    source = frame_bgr.copy()
    before_geometry = residual(source)
    geometry_image, geometry_report = correct_auto(source)
    after_geometry = residual(geometry_image)

    if enable_defringe:
        final_image, fringe_report = defringe(geometry_image)
    else:
        final_image = geometry_image.copy()
        fringe_report = {
            "changed": 0.0,
            "zone": 0.0,
            "candidate_pixels": 0,
            "corrected_pixels": 0,
            "changed_by_channel": {"b": 0, "g": 0, "r": 0},
            "mean_abs_delta": {"b": 0.0, "g": 0.0, "r": 0.0},
            "time": 0.0,
            "method": "disabled",
        }

    geometry_report_for_output = {
        key: value for key, value in geometry_report.items() if key != "models"
    }
    changed = np.any(final_image != source, axis=2)
    report: Dict[str, object] = {
        "geometry": geometry_report_for_output,
        "residual_before_geometry": before_geometry,
        "residual_after_geometry": after_geometry,
        "defringe": fringe_report,
        "changed_fraction_total": float(changed.mean()),
        "applied": bool(geometry_report["applied"] or fringe_report["corrected_pixels"]),
        "elapsed_seconds": 0.0,
    }
    report["elapsed_seconds"] = time.perf_counter() - started

    if return_stages:
        stages = {
            "source": source,
            "geometry": geometry_image,
            "final": final_image,
        }
        return final_image, report, stages
    return final_image, report
