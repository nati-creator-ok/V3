"""
Table detection module using object detection models.
Supports YOLOv8, Faster R-CNN, and other architectures.
"""

from .detector import TableDetector
from .yolo_detector import YOLOTableDetector
from .models import DetectionResult, BoundingBox

__all__ = [
    "TableDetector",
    "YOLOTableDetector", 
    "DetectionResult",
    "BoundingBox",
]
