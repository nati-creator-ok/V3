"""
Table Transformer-based table detector.

Uses Microsoft's Table Transformer model pretrained on PubTables-1M dataset
for accurate table detection in documents.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray

from .models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class TableTransformerDetector:
    """
    Detects tables using Microsoft's Table Transformer model.
    
    This model is specifically trained for table detection on the PubTables-1M dataset
    and provides much better accuracy than generic object detection models.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the Table Transformer detector.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu')
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Table Transformer model."""
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            
            logger.info("Loading Table Transformer model (microsoft/table-transformer-detection)...")
            
            # Load model and processor
            self.processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Table Transformer loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Table Transformer: {e}")
            raise
    
    def detect(self, image: NDArray) -> DetectionResult:
        """
        Detect tables in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            DetectionResult with detected table bounding boxes
        """
        import cv2
        
        h, w = image.shape[:2]
        
        # Convert BGR to RGB and then to PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([[h, w]])
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Convert to BoundingBox objects
        boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score_val = score.item()
            label_val = label.item()
            x1, y1, x2, y2 = box.tolist()
            
            # Label 0 is 'table' in this model
            class_name = "table" if label_val == 0 else f"class_{label_val}"
            
            # BoundingBox uses x, y, width, height format
            bbox = BoundingBox(
                x=float(x1),
                y=float(y1),
                width=float(x2 - x1),
                height=float(y2 - y1),
                confidence=float(score_val),
                class_name=class_name,
                class_id=int(label_val)
            )
            boxes.append(bbox)
        
        logger.info(f"Table Transformer detected {len(boxes)} tables")
        
        return DetectionResult(
            boxes=boxes,
            image_height=h,
            image_width=w
        )


def get_table_transformer_detector(
    device: str = "cpu",
    confidence_threshold: float = 0.5
) -> TableTransformerDetector:
    """Factory function to create a Table Transformer detector."""
    return TableTransformerDetector(
        device=device,
        confidence_threshold=confidence_threshold
    )
