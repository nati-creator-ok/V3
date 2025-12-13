"""
EasyOCR-based OCR engine implementation.
Provides good multilingual support for Korean and English.
"""

import logging
import time
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .engine import OCREngine
from .models import OCRResult, TextBox

logger = logging.getLogger(__name__)


class EasyOCREngine(OCREngine):
    """
    OCR engine using EasyOCR library.
    
    EasyOCR provides good accuracy for multilingual text recognition,
    especially for Korean and English mixed content.
    
    Example:
        engine = EasyOCREngine(languages=['ko', 'en'])
        result = engine.recognize(image)
        print(result.full_text)
    """
    
    # Language code mapping
    LANGUAGE_MAP = {
        "ko": "ko",
        "korean": "ko",
        "en": "en",
        "english": "en",
        "ja": "ja",
        "japanese": "ja",
        "zh": "ch_sim",
        "chinese": "ch_sim",
    }
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.0,
        paragraph: bool = False,
        batch_size: int = 1
    ):
        """
        Initialize EasyOCR engine.
        
        Args:
            languages: List of language codes
            device: Device for inference
            confidence_threshold: Minimum confidence threshold
            paragraph: Whether to merge text into paragraphs
            batch_size: Batch size for processing
        """
        super().__init__(
            languages=languages or ["ko", "en"],
            device=device,
            confidence_threshold=confidence_threshold
        )
        self.paragraph = paragraph
        self.batch_size = batch_size
        self.reader = None
    
    def initialize(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "easyocr package is required. Install with: pip install easyocr"
            )
        
        # Map language codes
        mapped_languages = []
        for lang in self.languages:
            mapped = self.LANGUAGE_MAP.get(lang.lower(), lang)
            if mapped not in mapped_languages:
                mapped_languages.append(mapped)
        
        gpu = self.device.startswith("cuda")
        
        logger.info(f"Initializing EasyOCR with languages: {mapped_languages}")
        self.reader = easyocr.Reader(
            mapped_languages,
            gpu=gpu,
            verbose=False
        )
        
        self._is_initialized = True
        logger.info("EasyOCR initialized successfully")
    
    def _recognize(
        self,
        image: NDArray[np.uint8],
        detail: bool = True
    ) -> OCRResult:
        """Run EasyOCR recognition."""
        start_time = time.time()
        h, w = image.shape[:2]
        
        # Run OCR with better accuracy settings
        results = self.reader.readtext(
            image,
            detail=1,  # Always get detailed results for processing
            paragraph=self.paragraph,
            batch_size=self.batch_size,
            width_ths=0.5,  # Better word separation
            height_ths=0.5,  # Better line detection
            low_text=0.3,  # Lower threshold for text detection
            text_threshold=0.5  # Text confidence threshold
        )
        
        inference_time = time.time() - start_time
        
        # Parse results
        text_boxes = []
        full_texts = []
        
        for result in results:
            if len(result) >= 2:
                bbox_points = result[0]  # List of 4 corner points
                text = result[1]
                confidence = result[2] if len(result) > 2 else 1.0
                
                # Convert polygon to bounding box
                points = np.array(bbox_points)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                text_boxes.append(TextBox(
                    text=text,
                    confidence=float(confidence),
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                ))
                full_texts.append(text)
        
        return OCRResult(
            text_boxes=text_boxes,
            full_text=" ".join(full_texts),
            inference_time=inference_time,
            engine_name="EasyOCR",
            languages=self.languages,
            image_width=w,
            image_height=h
        )
    
    def recognize_with_rotation(
        self,
        image: NDArray[np.uint8],
        rotations: List[int] = [0, 90, 180, 270]
    ) -> OCRResult:
        """
        Try OCR with different rotations and return best result.
        
        Args:
            image: Input image
            rotations: List of rotation angles to try
            
        Returns:
            OCRResult with highest average confidence
        """
        import cv2
        
        best_result = None
        best_confidence = -1
        
        for angle in rotations:
            # Rotate image
            if angle == 0:
                rotated = image
            elif angle == 90:
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # General rotation
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))
            
            result = self.recognize(rotated)
            
            if result.average_confidence > best_confidence:
                best_confidence = result.average_confidence
                best_result = result
                best_result.metadata["rotation"] = angle
        
        return best_result


class PaddleOCREngine(OCREngine):
    """
    OCR engine using PaddleOCR.
    Alternative engine with good performance for Asian languages.
    """
    
    LANGUAGE_MAP = {
        "ko": "korean",
        "en": "en",
        "zh": "ch",
        "ja": "japan",
    }
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.0
    ):
        super().__init__(
            languages=languages or ["ko", "en"],
            device=device,
            confidence_threshold=confidence_threshold
        )
        self.ocr = None
    
    def initialize(self) -> None:
        """Initialize PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr package is required. Install with: pip install paddleocr"
            )
        
        # PaddleOCR uses different language codes
        lang = self.LANGUAGE_MAP.get(self.languages[0], "en")
        use_gpu = self.device.startswith("cuda")
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
        
        self._is_initialized = True
        logger.info("PaddleOCR initialized successfully")
    
    def _recognize(
        self,
        image: NDArray[np.uint8],
        detail: bool = True
    ) -> OCRResult:
        """Run PaddleOCR recognition."""
        start_time = time.time()
        h, w = image.shape[:2]
        
        results = self.ocr.ocr(image, cls=True)
        
        inference_time = time.time() - start_time
        
        text_boxes = []
        full_texts = []
        
        if results and results[0]:
            for line in results[0]:
                bbox_points = line[0]
                text, confidence = line[1]
                
                # Convert to bounding box
                points = np.array(bbox_points)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                
                text_boxes.append(TextBox(
                    text=text,
                    confidence=float(confidence),
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                ))
                full_texts.append(text)
        
        return OCRResult(
            text_boxes=text_boxes,
            full_text=" ".join(full_texts),
            inference_time=inference_time,
            engine_name="PaddleOCR",
            languages=self.languages,
            image_width=w,
            image_height=h
        )
