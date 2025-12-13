"""
Data models for table detection results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Represents a bounding box for detected table region."""
    
    x: float  # Top-left x coordinate
    y: float  # Top-left y coordinate
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0  # 0 = table, could extend to other classes
    class_name: str = "table"
    
    @property
    def x1(self) -> float:
        """Left edge."""
        return self.x
    
    @property
    def y1(self) -> float:
        """Top edge."""
        return self.y
    
    @property
    def x2(self) -> float:
        """Right edge."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge."""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy)."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return as (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)
    
    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Return as (center_x, center_y, width, height) format."""
        cx, cy = self.center
        return (cx, cy, self.width, self.height)
    
    def scale(self, sx: float, sy: float) -> "BoundingBox":
        """Scale bounding box by given factors."""
        return BoundingBox(
            x=self.x * sx,
            y=self.y * sy,
            width=self.width * sx,
            height=self.height * sy,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name
        )
    
    def pad(self, padding: float) -> "BoundingBox":
        """Add padding to bounding box."""
        return BoundingBox(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name
        )
    
    def clip(self, max_width: float, max_height: float) -> "BoundingBox":
        """Clip bounding box to image boundaries."""
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(max_width, self.x2)
        y2 = min(max_height, self.y2)
        
        return BoundingBox(
            x=x1,
            y=y1,
            width=max(0, x2 - x1),
            height=max(0, y2 - y1),
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name
        )
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }
    
    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float = 1.0,
        class_id: int = 0,
        class_name: str = "table"
    ) -> "BoundingBox":
        """Create from (x1, y1, x2, y2) format."""
        return cls(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name
        )


@dataclass
class DetectionResult:
    """Results from table detection on a single image."""
    
    boxes: List[BoundingBox] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_time: float = 0.0  # seconds
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_tables(self) -> int:
        """Number of detected tables."""
        return len(self.boxes)
    
    def filter_by_confidence(self, min_confidence: float) -> "DetectionResult":
        """Return new result with boxes filtered by confidence threshold."""
        filtered_boxes = [b for b in self.boxes if b.confidence >= min_confidence]
        return DetectionResult(
            boxes=filtered_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time=self.inference_time,
            model_name=self.model_name,
            metadata=self.metadata
        )
    
    def filter_by_area(
        self,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None
    ) -> "DetectionResult":
        """Return new result with boxes filtered by area."""
        filtered_boxes = []
        for box in self.boxes:
            if min_area is not None and box.area < min_area:
                continue
            if max_area is not None and box.area > max_area:
                continue
            filtered_boxes.append(box)
        
        return DetectionResult(
            boxes=filtered_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time=self.inference_time,
            model_name=self.model_name,
            metadata=self.metadata
        )
    
    def apply_nms(self, iou_threshold: float = 0.5) -> "DetectionResult":
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if len(self.boxes) <= 1:
            return self
        
        # Sort by confidence
        sorted_boxes = sorted(self.boxes, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)
            
            sorted_boxes = [
                box for box in sorted_boxes
                if best.iou(box) < iou_threshold
            ]
        
        return DetectionResult(
            boxes=keep,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time=self.inference_time,
            model_name=self.model_name,
            metadata=self.metadata
        )
    
    def crop_tables(self, image: NDArray[np.uint8]) -> List[NDArray[np.uint8]]:
        """Crop detected table regions from image."""
        crops = []
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.to_xyxy())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2].copy())
        
        return crops
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_tables": self.num_tables,
            "boxes": [box.to_dict() for box in self.boxes],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "inference_time": self.inference_time,
            "model_name": self.model_name
        }
