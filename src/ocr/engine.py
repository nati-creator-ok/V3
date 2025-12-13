"""
Base class for OCR engines.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .models import OCRResult, TextBox

logger = logging.getLogger(__name__)


def get_device(device: str) -> str:
    """Auto-detect the best available device."""
    if device and device not in ("auto", ""):
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        return device
    
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class OCREngine(ABC):
    """
    Abstract base class for OCR engines.
    
    Subclasses should implement the _recognize method.
    """
    
    # Supported language codes
    SUPPORTED_LANGUAGES = ["en", "ko"]
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        device: str = "auto",
        confidence_threshold: float = 0.0
    ):
        """
        Initialize OCR engine.
        
        Args:
            languages: List of language codes (e.g., ['en', 'ko'])
            device: Device for inference ('cuda', 'cpu', 'auto')
            confidence_threshold: Minimum confidence for results
        """
        self.languages = languages or ["en"]
        self.device = get_device(device)
        self.confidence_threshold = confidence_threshold
        self._is_initialized = False
        logger.info(f"OCREngine initialized with device: {self.device}")
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the OCR engine. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _recognize(
        self,
        image: NDArray[np.uint8],
        detail: bool = True
    ) -> OCRResult:
        """
        Perform OCR on image.
        Must be implemented by subclasses.
        
        Args:
            image: Input image
            detail: Whether to return detailed results with bounding boxes
            
        Returns:
            OCRResult with recognized text
        """
        pass
    
    def recognize(
        self,
        image: NDArray[np.uint8],
        region: Optional[Tuple[int, int, int, int]] = None,
        detail: bool = True
    ) -> OCRResult:
        """
        Recognize text in image.
        
        Args:
            image: Input image (BGR format)
            region: Optional region to crop (x1, y1, x2, y2)
            detail: Whether to return detailed results
            
        Returns:
            OCRResult with recognized text
        """
        if not self._is_initialized:
            self.initialize()
        
        # Crop to region if specified
        if region is not None:
            x1, y1, x2, y2 = map(int, region)
            image = image[y1:y2, x1:x2].copy()
        
        result = self._recognize(image, detail=detail)
        
        # Filter by confidence
        if self.confidence_threshold > 0:
            result = result.filter_by_confidence(self.confidence_threshold)
        
        return result
    
    def recognize_batch(
        self,
        images: List[NDArray[np.uint8]],
        detail: bool = True
    ) -> List[OCRResult]:
        """
        Recognize text in multiple images.
        
        Args:
            images: List of input images
            detail: Whether to return detailed results
            
        Returns:
            List of OCRResult objects
        """
        results = []
        for image in images:
            result = self.recognize(image, detail=detail)
            results.append(result)
        return results
    
    def recognize_cells(
        self,
        image: NDArray[np.uint8],
        cell_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[str]:
        """
        Recognize text in multiple cell regions.
        
        Args:
            image: Full table image
            cell_bboxes: List of cell bounding boxes (x1, y1, x2, y2)
            
        Returns:
            List of recognized text strings
        """
        texts = []
        for bbox in cell_bboxes:
            result = self.recognize(image, region=bbox, detail=False)
            texts.append(result.full_text)
        return texts
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(languages={self.languages}, device={self.device})"
