"""
Main table extraction pipeline.
Orchestrates preprocessing, detection, structure recognition, and OCR.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .config import settings
from .preprocessing import PreprocessingPipeline, PreprocessingConfig
from .detection import YOLOTableDetector, DetectionResult
from .structure import TransformerTableRecognizer, TableStructure
from .ocr import EasyOCREngine, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from complete table extraction pipeline."""
    
    tables: List[TableStructure] = field(default_factory=list)
    detection_result: Optional[DetectionResult] = None
    ocr_results: List[OCRResult] = field(default_factory=list)
    preprocessing_info: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    page_number: int = 0
    source_file: Optional[str] = None
    image_size: tuple = (0, 0)
    
    @property
    def num_tables(self) -> int:
        return len(self.tables)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_tables": self.num_tables,
            "tables": [t.to_dict() for t in self.tables],
            "total_time": self.total_time,
            "page_number": self.page_number,
            "source_file": self.source_file,
            "image_size": list(self.image_size)
        }
    
    def to_csv(self, table_index: int = 0) -> str:
        """Export specific table to CSV."""
        if table_index < len(self.tables):
            return self.tables[table_index].to_csv()
        return ""
    
    def to_html(self, table_index: int = 0) -> str:
        """Export specific table to HTML."""
        if table_index < len(self.tables):
            return self.tables[table_index].to_html()
        return ""
    
    def to_excel(self, output_path: str, sheet_per_table: bool = True):
        """Export tables to Excel file."""
        import pandas as pd
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, table in enumerate(self.tables):
                df = table.to_dataframe()
                sheet_name = f"Table_{i + 1}" if sheet_per_table else "Tables"
                df.to_excel(writer, sheet_name=sheet_name, index=False)


class TableExtractionPipeline:
    """
    End-to-end pipeline for table extraction from documents.
    
    Example:
        pipeline = TableExtractionPipeline()
        results = pipeline.extract("document.pdf")
        
        for result in results:
            for table in result.tables:
                print(table.to_csv())
    """
    
    @staticmethod
    def _get_device(device: str) -> str:
        """Auto-detect the best available device."""
        if device and device != "auto":
            return device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def __init__(
        self,
        detector = None,
        structure_recognizer: Optional[TransformerTableRecognizer] = None,
        ocr_engine: Optional[EasyOCREngine] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        device: str = None,
        use_table_transformer: bool = True  # Use Table Transformer by default
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            detector: Table detection model (or None to create default)
            structure_recognizer: Structure recognition model
            ocr_engine: OCR engine for text extraction
            preprocessing_config: Preprocessing configuration
            device: Device for inference ('cuda', 'cpu', 'auto')
            use_table_transformer: Use Microsoft's Table Transformer (recommended)
        """
        self.device = self._get_device(device or settings.model_device)
        logger.info(f"Using device: {self.device}")
        
        # Initialize components with better preprocessing defaults
        if preprocessing_config is None:
            preprocessing_config = PreprocessingConfig(
                enable_deskew=True,
                enable_denoise=True,
                enable_contrast=True,
                enable_resolution_norm=True,
                denoise_strength=7,  # Moderate denoising
                clahe_clip_limit=3.0  # Better contrast
            )
        
        self.preprocessor = PreprocessingPipeline(
            config=preprocessing_config
        )
        
        # Use Table Transformer for better table detection (pretrained on tables)
        if detector is not None:
            self.detector = detector
        elif use_table_transformer:
            try:
                from .detection.table_transformer_detector import TableTransformerDetector
                self.detector = TableTransformerDetector(
                    device=self.device,
                    confidence_threshold=settings.confidence_threshold
                )
                logger.info("Using Table Transformer detector (pretrained on PubTables-1M)")
            except Exception as e:
                logger.warning(f"Failed to load Table Transformer, falling back to YOLO: {e}")
                self.detector = YOLOTableDetector(
                    model_path=settings.table_detection_weights,
                    device=self.device,
                    confidence_threshold=settings.confidence_threshold
                )
        else:
            self.detector = YOLOTableDetector(
            model_path=settings.table_detection_weights,
            device=self.device,
            confidence_threshold=settings.confidence_threshold
        )
        
        # Initialize Table Structure Extractor for accurate cell detection
        self.structure_extractor = None
        if use_table_transformer:
            try:
                from .detection.table_structure_extractor import TableStructureExtractor
                self.structure_extractor = TableStructureExtractor(
                    device=self.device,
                    confidence_threshold=settings.confidence_threshold
                )
                logger.info("Table Structure Extractor loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Table Structure Extractor: {e}")
                self.structure_extractor = None
        
        # Force HF structure extractor as the primary; disable custom recognizer by default
        self.structure_recognizer = structure_recognizer if structure_recognizer else None
        
        self.ocr_engine = ocr_engine or EasyOCREngine(
            languages=settings.ocr_languages,
            device=self.device
        )
    
    def extract(
        self,
        input_data: Union[str, Path, NDArray[np.uint8]],
        skip_preprocessing: bool = False,
        skip_ocr: bool = False
    ) -> ExtractionResult:
        """
        Extract tables from a single image.
        
        Args:
            input_data: Image path or numpy array
            skip_preprocessing: Skip preprocessing step
            skip_ocr: Skip OCR (return structure without text)
            
        Returns:
            ExtractionResult with extracted tables
        """
        start_time = time.time()
        
        # Load/preprocess image
        if isinstance(input_data, (str, Path)):
            source_file = str(input_data)
            if skip_preprocessing:
                import cv2
                image = cv2.imread(source_file)
                preprocessing_info = {}
            else:
                result = self.preprocessor.process(input_data)
                image = result.image
                preprocessing_info = {
                    "skew_angle": result.skew_angle,
                    "transforms": result.applied_transforms
                }
        else:
            source_file = None
            if skip_preprocessing:
                image = input_data
                preprocessing_info = {}
            else:
                result = self.preprocessor.process(input_data)
                image = result.image
                preprocessing_info = {
                    "skew_angle": result.skew_angle,
                    "transforms": result.applied_transforms
                }
        
        h, w = image.shape[:2]
        
        # Detect tables
        detection_result = self.detector.detect(image)
        logger.info(f"Table detector found {len(detection_result.boxes)} table(s)")
        
        # If YOLO didn't find tables, try line-based detection
        if not detection_result.boxes:
            logger.info("No tables detected, trying line-based detection...")
            try:
                from .detection.line_detector import LineBasedTableDetector
                line_detector = LineBasedTableDetector()
                detection_result = line_detector.detect(image)
                if detection_result.boxes:
                    logger.info(f"Line-based detector found {len(detection_result.boxes)} tables")
            except Exception as e:
                logger.warning(f"Line-based detection failed: {e}")
        
        # Process each detected table AND extract standalone text
        tables = []
        ocr_results = []
        standalone_text_regions = []
        
        for box in detection_result.boxes:
            # Crop table region
            x1, y1, x2, y2 = map(int, box.to_xyxy())
            # Ensure valid crop bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid table bounds: {(x1, y1, x2, y2)}")
                continue
                
            table_image = image[y1:y2, x1:x2].copy()
            table_h, table_w = table_image.shape[:2]
            
            if table_h < 10 or table_w < 10:
                logger.warning(f"Table too small: {(table_w, table_h)}")
                continue
            
            # Extract table structure using Table Transformer structure recognition
            try:
                if self.structure_extractor:
                    # Use proper structure extraction model
                    table = self.structure_extractor.extract_structure(
                        table_image,
                        table_bbox=(x1, y1, x2, y2)
                    )
                    
                    if table and table.cells:
                        # Run OCR and populate cells with text
                        if not skip_ocr:
                            try:
                                ocr_result = self.ocr_engine.recognize(table_image)
                                ocr_results.append(ocr_result)
                                
                                # Assign OCR text to detected cells
                                for cell in table.cells:
                                    if cell.bbox:
                                        # Get text within cell bounds
                                        cell_x1, cell_y1, cell_x2, cell_y2 = cell.bbox
                                        cell_texts = []
                                        
                                        for text_box in ocr_result.text_boxes:
                                            if text_box.bbox:
                                                tx1, ty1, tx2, ty2 = text_box.bbox
                                                # Check if text box overlaps with cell
                                                if not (tx2 < cell_x1 or tx1 > cell_x2 or ty2 < cell_y1 or ty1 > cell_y2):
                                                    cell_texts.append(text_box.text)
                                        
                                        cell.text = " ".join(cell_texts).strip()
                                
                                logger.info(f"Extracted table with {len(table.cells)} cells ({table.num_rows} rows x {table.num_cols} cols)")
                                tables.append(table)
                            except Exception as e:
                                logger.warning(f"OCR failed for table: {e}")
                                tables.append(table)  # Add table even without text
                        else:
                            tables.append(table)
                    else:
                        logger.warning("Structure extractor found no cells")
                else:
                    # Fallback to OCR-based structure extraction
                    logger.info("Using fallback OCR-based structure extraction")
                    if not skip_ocr:
                        ocr_result = self.ocr_engine.recognize(table_image)
                        ocr_results.append(ocr_result)
                        
                        if ocr_result.text_boxes:
                            table = self._create_table_from_ocr(ocr_result, (table_h, table_w))
                            if table:
                                table.bbox = (x1, y1, x2, y2)
                                tables.append(table)
                                logger.info(f"Created table with {len(table.cells)} cells from OCR")
            except Exception as e:
                logger.error(f"Table structure extraction failed: {e}", exc_info=True)
            else:
                # Skip custom recognizer fallback when HF extractor is unavailable
                logger.info("Skipping custom transformer recognizer; relying on OCR fallback only.")
        
        # Always extract standalone text from non-table regions
        if not skip_ocr:
            logger.info("Extracting text from non-table regions...")
            try:
                # Get regions that are NOT tables
                table_masks = []
                for box in detection_result.boxes:
                    x1, y1, x2, y2 = map(int, box.to_xyxy())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    table_masks.append((x1, y1, x2, y2))
                
                # Run OCR on full image to get all text
                full_ocr = self.ocr_engine.recognize(image)
                ocr_results.append(full_ocr)
                
                # Filter out text boxes that fall within table regions
                standalone_boxes = []
                for text_box in full_ocr.text_boxes:
                    if text_box.bbox:
                        box_center_x = (text_box.bbox[0] + text_box.bbox[2]) / 2
                        box_center_y = (text_box.bbox[1] + text_box.bbox[3]) / 2
                        
                        # Check if box center is inside any table region
                        in_table = False
                        for tx1, ty1, tx2, ty2 in table_masks:
                            if tx1 <= box_center_x <= tx2 and ty1 <= box_center_y <= ty2:
                                in_table = True
                                break
                        
                        if not in_table:
                            standalone_boxes.append(text_box)
                
                # Create a "standalone text" table from non-table text
                if standalone_boxes:
                    from .ocr.models import OCRResult
                    standalone_ocr = OCRResult(
                        text_boxes=standalone_boxes,
                        full_text=" ".join([b.text for b in standalone_boxes]),
                        language=full_ocr.language
                    )
                    standalone_table = self._create_table_from_ocr(standalone_ocr, (h, w))
                    if standalone_table:
                        standalone_table.bbox = (0, 0, w, h)
                        tables.append(standalone_table)
                        logger.info(f"Extracted {len(standalone_boxes)} standalone text regions")
                
            except Exception as e:
                logger.warning(f"Standalone text extraction failed: {e}")
        
        total_time = time.time() - start_time
        
        return ExtractionResult(
            tables=tables,
            detection_result=detection_result,
            ocr_results=ocr_results,
            preprocessing_info=preprocessing_info,
            total_time=total_time,
            source_file=source_file,
            image_size=(h, w)
        )
    
    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None
    ) -> List[ExtractionResult]:
        """
        Extract tables from PDF document.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to process (0-indexed), or None for all
            
        Returns:
            List of ExtractionResult, one per page
        """
        # Process PDF pages
        page_results = self.preprocessor.process_pdf(pdf_path)
        
        if pages is not None:
            page_results = [page_results[i] for i in pages if i < len(page_results)]
        
        results = []
        for idx, preprocess_result in enumerate(page_results):
            # Extract from preprocessed image
            result = self.extract(
                preprocess_result.image,
                skip_preprocessing=True  # Already preprocessed
            )
            result.page_number = preprocess_result.metadata.get("page_number", idx)
            result.source_file = str(pdf_path)
            result.preprocessing_info = {
                "skew_angle": preprocess_result.skew_angle,
                "transforms": preprocess_result.applied_transforms
            }
            results.append(result)
        
        return results
    
    def extract_batch(
        self,
        inputs: List[Union[str, Path, NDArray[np.uint8]]]
    ) -> List[ExtractionResult]:
        """
        Extract tables from multiple images.
        
        Args:
            inputs: List of image paths or arrays
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        for input_data in inputs:
            try:
                result = self.extract(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                results.append(ExtractionResult())
        return results
    
    def _assign_text_to_cells(
        self,
        table: TableStructure,
        ocr_result: OCRResult
    ) -> None:
        """Assign OCR text to table cells based on bounding box overlap."""
        for cell in table.cells:
            if cell.bbox is None:
                continue
            
            # Get text that falls within cell region
            cell_text = ocr_result.get_text_in_region(
                cell.bbox,
                iou_threshold=0.3
            )
            cell.text = cell_text.strip()
            
            # Calculate confidence based on OCR confidence
            relevant_boxes = [
                tb for tb in ocr_result.text_boxes
                if self._box_overlap_ratio(cell.bbox, tb.bbox) > 0.3
            ]
            if relevant_boxes:
                cell.confidence = sum(tb.confidence for tb in relevant_boxes) / len(relevant_boxes)
    
    def _create_table_from_ocr(
        self,
        ocr_result: OCRResult,
        image_size: tuple
    ) -> Optional[TableStructure]:
        """
        Create a simple table structure from OCR results by grouping text by rows.
        
        This is a fallback when no tables are detected by the detection model.
        Uses smart grid alignment to properly detect columns.
        """
        from .structure.models import Cell
        
        if not ocr_result.text_boxes:
            return None
        
        # Filter boxes with valid bboxes
        valid_boxes = [b for b in ocr_result.text_boxes if b.bbox]
        if not valid_boxes:
            return None
        
        # Sort text boxes by y-coordinate (top to bottom)
        sorted_boxes = sorted(valid_boxes, key=lambda b: b.bbox[1])
        
        # Calculate average character height to determine row threshold
        heights = [b.bbox[3] - b.bbox[1] for b in sorted_boxes if b.bbox]
        avg_height = sum(heights) / len(heights) if heights else 30
        row_height_threshold = avg_height * 0.6
        
        # Group boxes into rows based on y-coordinate proximity
        rows = []
        current_row = [sorted_boxes[0]]
        row_y = sorted_boxes[0].bbox[1]
        
        for box in sorted_boxes[1:]:
            box_y = box.bbox[1]
            # Check if this box is on the same row (y-coordinates overlap significantly)
            if abs(box_y - row_y) < row_height_threshold:
                current_row.append(box)
            else:
                # Sort current row by x-coordinate (left to right)
                current_row.sort(key=lambda b: b.bbox[0])
                rows.append(current_row)
                current_row = [box]
                row_y = box_y
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda b: b.bbox[0])
            rows.append(current_row)
        
        # Detect column boundaries using clustering of x-coordinates
        all_x_starts = []
        for row in rows:
            for box in row:
                all_x_starts.append(box.bbox[0])
        
        if not all_x_starts:
            return None
        
        # Cluster x-coordinates to find column boundaries
        all_x_starts.sort()
        col_boundaries = [all_x_starts[0]]
        min_col_gap = avg_height * 0.5  # Minimum gap between columns
        
        for x in all_x_starts[1:]:
            if x - col_boundaries[-1] > min_col_gap:
                col_boundaries.append(x)
        
        num_cols = len(col_boundaries)
        
        # Function to find which column a box belongs to
        def get_column(box_x):
            for i, boundary in enumerate(col_boundaries):
                if i == len(col_boundaries) - 1:
                    return i
                next_boundary = col_boundaries[i + 1]
                if box_x < (boundary + next_boundary) / 2:
                    return i
            return len(col_boundaries) - 1
        
        # Create cells from rows with proper column alignment
        cells = []
        for row_idx, row in enumerate(rows):
            row_cells = {}  # Track which columns are filled in this row
            for box in row:
                col_idx = get_column(box.bbox[0])
                # If column already has content, append to it
                if col_idx in row_cells:
                    row_cells[col_idx].text += " " + box.text
                else:
                    row_cells[col_idx] = Cell(
                        row=row_idx,
                        col=col_idx,
                        rowspan=1,
                        colspan=1,
                        bbox=box.bbox,
                        text=box.text,
                        confidence=box.confidence
                    )
            
            # Add all cells from this row
            cells.extend(row_cells.values())
        
        h, w = image_size
        table = TableStructure(
            cells=cells,
            num_rows=len(rows),
            num_cols=num_cols,
            bbox=(0, 0, w, h),
            confidence=ocr_result.average_confidence
        )
        
        return table
    
    @staticmethod
    def _box_overlap_ratio(
        box1: tuple,
        box2: tuple
    ) -> float:
        """Calculate overlap ratio between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / box2_area if box2_area > 0 else 0.0


def create_pipeline(
    device: str = "cuda",
    detection_weights: Optional[str] = None,
    structure_weights: Optional[str] = None,
    ocr_languages: List[str] = None
) -> TableExtractionPipeline:
    """
    Factory function to create configured pipeline.
    
    Args:
        device: Device for inference
        detection_weights: Path to detection model weights
        structure_weights: Path to structure model weights
        ocr_languages: Languages for OCR
        
    Returns:
        Configured TableExtractionPipeline
    """
    detector = YOLOTableDetector(
        model_path=detection_weights,
        device=device
    )
    
    structure_recognizer = TransformerTableRecognizer(
        model_path=structure_weights,
        device=device
    )
    
    ocr_engine = EasyOCREngine(
        languages=ocr_languages or ["ko", "en"],
        device=device
    )
    
    return TableExtractionPipeline(
        detector=detector,
        structure_recognizer=structure_recognizer,
        ocr_engine=ocr_engine,
        device=device
    )
