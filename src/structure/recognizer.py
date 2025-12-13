"""
Base class for table structure recognition models.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .models import TableStructure, StructureRecognitionResult

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


class TableStructureRecognizer(ABC):
    """
    Abstract base class for table structure recognition.
    
    Subclasses should implement the _recognize method to detect
    rows, columns, cells, and merged cell regions.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize structure recognizer.
        
        Args:
            model_path: Path to model weights
            device: Device to run inference on ('cuda', 'cpu', 'auto')
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = get_device(device)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._is_loaded = False
        logger.info(f"TableStructureRecognizer initialized with device: {self.device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _recognize(
        self,
        image: NDArray[np.uint8],
        table_bbox: Optional[tuple] = None
    ) -> StructureRecognitionResult:
        """
        Recognize table structure in image.
        Must be implemented by subclasses.
        
        Args:
            image: Input image (cropped table region or full page)
            table_bbox: Optional bounding box of table in image
            
        Returns:
            StructureRecognitionResult with detected table structure
        """
        pass
    
    def recognize(
        self,
        image: NDArray[np.uint8],
        table_bbox: Optional[tuple] = None
    ) -> StructureRecognitionResult:
        """
        Recognize table structure.
        
        Args:
            image: Input image
            table_bbox: Optional table bounding box (x1, y1, x2, y2)
            
        Returns:
            StructureRecognitionResult
        """
        if not self._is_loaded:
            self.load_model()
        
        # Crop to table region if bbox provided
        if table_bbox is not None:
            x1, y1, x2, y2 = map(int, table_bbox)
            image = image[y1:y2, x1:x2].copy()
        
        result = self._recognize(image, table_bbox)
        
        # Filter low confidence cells
        result.table.cells = [
            cell for cell in result.table.cells
            if cell.confidence >= self.confidence_threshold
        ]
        
        return result
    
    def recognize_batch(
        self,
        images: List[NDArray[np.uint8]],
        table_bboxes: Optional[List[tuple]] = None
    ) -> List[StructureRecognitionResult]:
        """
        Recognize structure in multiple table images.
        
        Args:
            images: List of input images
            table_bboxes: Optional list of table bounding boxes
            
        Returns:
            List of StructureRecognitionResult objects
        """
        if table_bboxes is None:
            table_bboxes = [None] * len(images)
        
        results = []
        for image, bbox in zip(images, table_bboxes):
            result = self.recognize(image, bbox)
            results.append(result)
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
