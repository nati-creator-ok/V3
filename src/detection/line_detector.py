"""
Line-based table detector using OpenCV.

Detects tables by finding horizontal and vertical lines in the image,
then identifying rectangular regions bounded by these lines.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class LineBasedTableDetector:
    """
    Detects tables in images using line detection.
    
    This detector works by:
    1. Finding horizontal and vertical lines using morphological operations
    2. Finding intersection points of these lines
    3. Identifying rectangular regions as potential tables
    """
    
    def __init__(
        self,
        min_line_length: int = 50,
        line_thickness: int = 2,
        min_table_area: int = 5000,
        confidence: float = 0.8
    ):
        """
        Initialize line-based detector.
        
        Args:
            min_line_length: Minimum length for detected lines
            line_thickness: Expected thickness of table lines
            min_table_area: Minimum area for a detected table region
            confidence: Confidence score for detected tables
        """
        self.min_line_length = min_line_length
        self.line_thickness = line_thickness
        self.min_table_area = min_table_area
        self.confidence = confidence
    
    def detect(self, image: NDArray) -> DetectionResult:
        """
        Detect tables in an image using line detection.
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            DetectionResult with detected table bounding boxes
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (max(40, w // 30), 1)
        )
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines  
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(40, h // 30))
        )
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.add(horizontal, vertical)
        
        # Dilate to connect nearby lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_mask = cv2.dilate(table_mask, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(
            table_mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_table_area:
                continue
            
            # Get bounding rectangle
            x, y, rect_w, rect_h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (tables are usually not too narrow)
            aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            # Check if the region has enough lines inside
            roi_lines = table_mask[y:y+rect_h, x:x+rect_w]
            line_density = np.sum(roi_lines > 0) / (rect_w * rect_h) if rect_w * rect_h > 0 else 0
            
            # Tables typically have line density between 1% and 30%
            if line_density < 0.01 or line_density > 0.3:
                continue
            
            box = BoundingBox(
                x=float(x),
                y=float(y),
                width=float(rect_w),
                height=float(rect_h),
                confidence=self.confidence,
                class_name="table",
                class_id=0
            )
            boxes.append(box)
        
        # Sort by area (largest first)
        boxes.sort(key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1), reverse=True)
        
        # Non-maximum suppression to remove overlapping boxes
        boxes = self._nms(boxes, iou_threshold=0.5)
        
        logger.info(f"Line-based detector found {len(boxes)} tables")
        
        return DetectionResult(
            boxes=boxes,
            image_height=h,
            image_width=w
        )
    
    def _nms(
        self, 
        boxes: List[BoundingBox], 
        iou_threshold: float = 0.5
    ) -> List[BoundingBox]:
        """Apply non-maximum suppression."""
        if not boxes:
            return boxes
        
        keep = []
        boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        while boxes_sorted:
            best = boxes_sorted.pop(0)
            keep.append(best)
            
            boxes_sorted = [
                box for box in boxes_sorted
                if self._iou(best, box) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def detect_tables_by_lines(image: NDArray) -> List[Tuple[int, int, int, int]]:
    """
    Simple function to detect table regions in an image.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        List of bounding boxes as (x1, y1, x2, y2) tuples
    """
    detector = LineBasedTableDetector()
    result = detector.detect(image)
    return [(int(b.x1), int(b.y1), int(b.x2), int(b.y2)) for b in result.boxes]
