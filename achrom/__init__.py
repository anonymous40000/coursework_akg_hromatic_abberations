"""Microscope chromatic-aberration correction package."""

from .auto import correct_auto, residual
from .defringe import defringe
from .pipeline import process_frame

__all__ = ["correct_auto", "residual", "defringe", "process_frame"]
