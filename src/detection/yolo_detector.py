"""
YOLOv8-based table detector implementation.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .detector import TableDetector
from .models import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


class YOLOTableDetector(TableDetector):
    """
    Table detector using YOLOv8 architecture.
    
    Example:
        detector = YOLOTableDetector(model_path="weights/table_detector.pt")
        result = detector.detect(image)
        for box in result.boxes:
            print(f"Table at {box.to_xyxy()} with confidence {box.confidence:.2f}")
    """
    
    # Class names for table detection
    CLASS_NAMES = {
        0: "table",
        1: "table_rotated",  # Optional: for rotated tables
    }
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        image_size: int = 640
    ):
        """
        Initialize YOLOv8 table detector.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            device: Device for inference ('cuda', 'cpu', 'mps')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            image_size: Input image size for YOLO
        """
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        self.image_size = image_size
    
    def load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLOv8. "
                "Install with: pip install ultralytics"
            )
        
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
        else:
            # Use pretrained model or initialize for training
            logger.warning("No model path provided, using YOLOv8n pretrained model")
            self.model = YOLO("yolov8n.pt")
        
        # Set device
        self.model.to(self.device)
        self._is_loaded = True
        logger.info(f"YOLO model loaded on device: {self.device}")
    
    def _detect(self, image: NDArray[np.uint8]) -> DetectionResult:
        """
        Run YOLOv8 inference on image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            DetectionResult with detected tables
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        # Parse results
        boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i].cpu().numpy())
                cls = int(result.boxes.cls[i].cpu().numpy())
                
                x1, y1, x2, y2 = xyxy
                class_name = self.CLASS_NAMES.get(cls, "table")
                
                box = BoundingBox.from_xyxy(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=conf,
                    class_id=cls,
                    class_name=class_name
                )
                boxes.append(box)
        
        h, w = image.shape[:2]
        
        return DetectionResult(
            boxes=boxes,
            image_width=w,
            image_height=h,
            inference_time=inference_time,
            model_name="YOLOv8"
        )
    
    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        pretrained: bool = True,
        **kwargs
    ) -> dict:
        """
        Train YOLO model on table detection dataset.
        
        Args:
            data_yaml: Path to data configuration YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            image_size: Input image size
            pretrained: Whether to use pretrained weights
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        if not self._is_loaded:
            self.load_model()
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            pretrained=pretrained,
            device=self.device,
            **kwargs
        )
        
        return results
    
    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export model to different formats for deployment.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tensorrt', etc.)
            output_path: Output path for exported model
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        if not self._is_loaded:
            self.load_model()
        
        export_path = self.model.export(format=format, **kwargs)
        
        if output_path:
            import shutil
            shutil.move(export_path, output_path)
            return output_path
        
        return export_path


class DetectronTableDetector(TableDetector):
    """
    Table detector using Detectron2 (Faster R-CNN / Mask R-CNN).
    Placeholder implementation - requires detectron2 installation.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        config_file: Optional[str] = None
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        self.config_file = config_file
    
    def load_model(self) -> None:
        """Load Detectron2 model."""
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
        except ImportError:
            raise ImportError(
                "detectron2 is required for this detector. "
                "Install following: https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
            )
        
        cfg = get_cfg()
        
        if self.config_file:
            cfg.merge_from_file(self.config_file)
        else:
            # Use default Faster R-CNN config
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            )
        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device
        
        if self.model_path and Path(self.model_path).exists():
            cfg.MODEL.WEIGHTS = str(self.model_path)
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            )
        
        # Set number of classes (1 for table)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        self.predictor = DefaultPredictor(cfg)
        self._is_loaded = True
        logger.info("Detectron2 model loaded")
    
    def _detect(self, image: NDArray[np.uint8]) -> DetectionResult:
        """Run Detectron2 inference."""
        start_time = time.time()
        
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        inference_time = time.time() - start_time
        
        boxes = []
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        for i in range(len(pred_boxes)):
            x1, y1, x2, y2 = pred_boxes[i]
            box = BoundingBox.from_xyxy(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(scores[i]),
                class_id=int(classes[i]),
                class_name="table"
            )
            boxes.append(box)
        
        h, w = image.shape[:2]
        
        return DetectionResult(
            boxes=boxes,
            image_width=w,
            image_height=h,
            inference_time=inference_time,
            model_name="Detectron2"
        )
