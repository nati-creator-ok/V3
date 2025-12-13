"""
OCR module for text extraction from table cells.
Supports multiple OCR engines with Korean/English multilingual support.
"""

from .engine import OCREngine
from .easyocr_engine import EasyOCREngine
from .tesseract_engine import TesseractEngine
from .models import OCRResult, TextBox

__all__ = [
    "OCREngine",
    "EasyOCREngine",
    "TesseractEngine",
    "OCRResult",
    "TextBox",
]
