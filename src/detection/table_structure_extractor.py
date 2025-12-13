"""
Table Structure Extraction using Microsoft's Table Transformer.

Extracts the cell-level structure of detected tables using the
table-transformer-structure-recognition model.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray

from ..structure.models import Cell, TableStructure

logger = logging.getLogger(__name__)


class TableStructureExtractor:
    """
    Extracts table structure (rows, columns, cells) using Table Transformer.
    
    This uses Microsoft's table-transformer-structure-recognition model
    which is trained to detect individual table cells and their structure.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the structure extractor.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu')
            confidence_threshold: Minimum confidence for cell detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Table Transformer structure recognition model."""
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            
            logger.info("Loading Table Structure Recognition model...")
            
            # Load model and processor
            self.processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Table Structure model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Table Structure model: {e}")
            raise
    
    def extract_structure(
        self,
        image: NDArray,
        table_bbox: Tuple[int, int, int, int] = None
    ) -> TableStructure:
        """
        Extract table structure from an image region.
        
        Args:
            image: Input image (table region) as numpy array (BGR format)
            table_bbox: Bounding box of table in original image (x1, y1, x2, y2)
            
        Returns:
            TableStructure with detected cells
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
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Extract cells
        cells = []
        rows_detected = set()
        cols_detected = set()
        
        # Group detections by type
        cell_boxes = []
        row_boxes = []
        col_boxes = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = score.item()
            label = label.item()
            box = box.tolist()
            
            # Get label name
            label_name = self.model.config.id2label[label]
            
            if label_name == "table column":
                col_boxes.append(box)
            elif label_name == "table row":
                row_boxes.append(box)
            elif label_name == "table" or label_name.startswith("table"):
                # Sometimes cells are labeled as "table"
                cell_boxes.append((box, score))
        
        # Sort rows and columns
        row_boxes.sort(key=lambda b: b[1])  # Sort by y1
        col_boxes.sort(key=lambda b: b[0])  # Sort by x1
        
        logger.info(f"Detected {len(row_boxes)} rows, {len(col_boxes)} columns")
        
        # Create cells from row/column intersections
        if row_boxes and col_boxes:
            for row_idx, row_box in enumerate(row_boxes):
                for col_idx, col_box in enumerate(col_boxes):
                    # Calculate intersection
                    x1 = max(col_box[0], 0)
                    y1 = max(row_box[1], 0)
                    x2 = min(col_box[2], w)
                    y2 = min(row_box[3], h)
                    
                    if x2 > x1 and y2 > y1:
                        # Valid cell
                        cell = Cell(
                            row=row_idx,
                            col=col_idx,
                            rowspan=1,
                            colspan=1,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            text="",
                            confidence=0.9
                        )
                        cells.append(cell)
        elif cell_boxes:
            # Fallback: use detected cells directly
            logger.info(f"Using {len(cell_boxes)} directly detected cells")
            
            # Sort cells by position (top-to-bottom, left-to-right)
            cell_boxes.sort(key=lambda item: (item[0][1], item[0][0]))
            
            # Group into rows based on y-position
            rows = []
            current_row = []
            current_y = cell_boxes[0][0][1] if cell_boxes else 0
            row_threshold = 20  # pixels
            
            for box, score in cell_boxes:
                x1, y1, x2, y2 = box
                
                if abs(y1 - current_y) > row_threshold:
                    if current_row:
                        rows.append(sorted(current_row, key=lambda b: b[0][0]))
                    current_row = [(box, score)]
                    current_y = y1
                else:
                    current_row.append((box, score))
            
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b[0][0]))
            
            # Create cells
            for row_idx, row in enumerate(rows):
                for col_idx, (box, score) in enumerate(row):
                    x1, y1, x2, y2 = map(int, box)
                    cell = Cell(
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        bbox=(x1, y1, x2, y2),
                        text="",
                        confidence=score
                    )
                    cells.append(cell)
            
            logger.info(f"Created {len(cells)} cells from {len(rows)} rows")
        
        # Determine table dimensions
        num_rows = max((c.row for c in cells), default=0) + 1 if cells else 0
        num_cols = max((c.col for c in cells), default=0) + 1 if cells else 0
        
        return TableStructure(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            bbox=table_bbox or (0, 0, w, h),
            confidence=0.9 if cells else 0.0
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
