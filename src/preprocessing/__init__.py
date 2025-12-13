"""
Preprocessing module for document images.
Handles deskewing, denoising, binarization, and normalization.
"""

from .pipeline import PreprocessingPipeline, PreprocessingConfig, PreprocessingResult
from .transforms import (
    adaptive_binarize,
    correct_skew,
    denoise,
    enhance_contrast,
    normalize_resolution,
    remove_background,
)

__all__ = [
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "PreprocessingResult",
    "correct_skew",
    "denoise",
    "adaptive_binarize",
    "enhance_contrast",
    "normalize_resolution",
    "remove_background",
]
