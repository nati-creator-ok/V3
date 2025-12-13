"""
Data models for OCR results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TextBox:
    """Represents a detected text region with its content."""
    
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    language: Optional[str] = None
    
    @property
    def x1(self) -> float:
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBox":
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            bbox=tuple(data["bbox"]),
            language=data.get("language")
        )


@dataclass
class OCRResult:
    """Result from OCR processing."""
    
    text_boxes: List[TextBox] = field(default_factory=list)
    full_text: str = ""
    inference_time: float = 0.0
    engine_name: str = ""
    languages: List[str] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_detections(self) -> int:
        return len(self.text_boxes)
    
    @property
    def average_confidence(self) -> float:
        if not self.text_boxes:
            return 0.0
        return sum(tb.confidence for tb in self.text_boxes) / len(self.text_boxes)
    
    def filter_by_confidence(self, min_confidence: float) -> "OCRResult":
        """Return new result with text boxes filtered by confidence."""
        filtered = [tb for tb in self.text_boxes if tb.confidence >= min_confidence]
        return OCRResult(
            text_boxes=filtered,
            full_text=" ".join(tb.text for tb in filtered),
            inference_time=self.inference_time,
            engine_name=self.engine_name,
            languages=self.languages,
            image_width=self.image_width,
            image_height=self.image_height,
            metadata=self.metadata
        )
    
    def get_text_in_region(
        self,
        bbox: Tuple[float, float, float, float],
        iou_threshold: float = 0.5
    ) -> str:
        """Get concatenated text from boxes that overlap with given region."""
        x1, y1, x2, y2 = bbox
        texts = []
        
        for tb in self.text_boxes:
            # Calculate intersection
            ix1 = max(x1, tb.x1)
            iy1 = max(y1, tb.y1)
            ix2 = min(x2, tb.x2)
            iy2 = min(y2, tb.y2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            
            intersection = (ix2 - ix1) * (iy2 - iy1)
            tb_area = tb.width * tb.height
            
            if tb_area > 0 and intersection / tb_area >= iou_threshold:
                texts.append(tb.text)
        
        return " ".join(texts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_boxes": [tb.to_dict() for tb in self.text_boxes],
            "full_text": self.full_text,
            "inference_time": self.inference_time,
            "engine_name": self.engine_name,
            "languages": self.languages,
            "num_detections": self.num_detections,
            "average_confidence": self.average_confidence
        }
