"""
Base class for table detection models.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


def get_device(device: str) -> str:
    """Auto-detect the best available device."""
    if device and device not in ("auto", ""):
        # If explicitly set to cuda, check if it's available
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


class TableDetector(ABC):
    """
    Abstract base class for table detection models.
    
    Subclasses should implement the _detect method.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ):
        """
        Initialize table detector.
        
        Args:
            model_path: Path to model weights
            device: Device to run inference on ('cuda', 'cpu', 'mps', 'auto')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = get_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._is_loaded = False
        logger.info(f"TableDetector initialized with device: {self.device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _detect(self, image: NDArray[np.uint8]) -> DetectionResult:
        """
        Perform detection on a single image.
        Must be implemented by subclasses.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            DetectionResult with detected table bounding boxes
        """
        pass
    
    def detect(
        self,
        image: NDArray[np.uint8],
        apply_nms: bool = True
    ) -> DetectionResult:
        """
        Detect tables in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            apply_nms: Whether to apply Non-Maximum Suppression
            
        Returns:
            DetectionResult with detected table bounding boxes
        """
        if not self._is_loaded:
            self.load_model()
        
        # Run detection
        result = self._detect(image)
        
        # Filter by confidence
        result = result.filter_by_confidence(self.confidence_threshold)
        
        # Apply NMS if requested
        if apply_nms:
            result = result.apply_nms(self.iou_threshold)
        
        return result
    
    def detect_batch(
        self,
        images: List[NDArray[np.uint8]],
        apply_nms: bool = True
    ) -> List[DetectionResult]:
        """
        Detect tables in multiple images.
        
        Args:
            images: List of input images
            apply_nms: Whether to apply Non-Maximum Suppression
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for image in images:
            result = self.detect(image, apply_nms=apply_nms)
            results.append(result)
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
