"""
Evaluation runner for table extraction system.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.pipeline import TableExtractionPipeline, ExtractionResult
from .metrics import (
    EvaluationReport,
    calculate_cell_f1,
    calculate_teds,
    calculate_cer,
    calculate_wer,
    calculate_detection_metrics,
)

logger = logging.getLogger(__name__)


class TableExtractionEvaluator:
    """
    Evaluator for table extraction system.
    
    Example:
        evaluator = TableExtractionEvaluator(pipeline)
        report = evaluator.evaluate(test_dataset)
        print(report)
    """
    
    def __init__(
        self,
        pipeline: Optional[TableExtractionPipeline] = None,
        iou_threshold: float = 0.5
    ):
        """
        Initialize evaluator.
        
        Args:
            pipeline: Table extraction pipeline
            iou_threshold: IoU threshold for matching
        """
        self.pipeline = pipeline
        self.iou_threshold = iou_threshold
    
    def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        verbose: bool = True
    ) -> EvaluationReport:
        """
        Evaluate pipeline on dataset.
        
        Args:
            dataset: List of samples with 'image_path', 'tables' (ground truth)
            verbose: Print progress
            
        Returns:
            EvaluationReport with all metrics
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not set")
        
        per_sample_results = []
        all_detection_metrics = []
        all_cell_f1 = []
        all_teds = []
        all_cer = []
        all_wer = []
        all_times = []
        
        for idx, sample in enumerate(dataset):
            if verbose:
                logger.info(f"Evaluating sample {idx + 1}/{len(dataset)}")
            
            try:
                # Run extraction
                result = self.pipeline.extract(sample['image_path'])
                all_times.append(result.total_time)
                
                # Get ground truth
                gt_tables = sample.get('tables', [])
                
                # Evaluate detection
                pred_boxes = [tuple(t.bbox) for t in result.tables if t.bbox]
                gt_boxes = [tuple(t['bbox']) for t in gt_tables if t.get('bbox')]
                
                det_metrics = calculate_detection_metrics(
                    pred_boxes, gt_boxes, self.iou_threshold
                )
                all_detection_metrics.append(det_metrics)
                
                # Evaluate structure per table
                for t_idx, (pred_table, gt_table) in enumerate(
                    zip(result.tables, gt_tables)
                ):
                    # Cell F1
                    pred_cells = [c.to_dict() for c in pred_table.cells]
                    gt_cells = gt_table.get('cells', [])
                    
                    prec, rec, f1 = calculate_cell_f1(
                        pred_cells, gt_cells, self.iou_threshold
                    )
                    all_cell_f1.append({'precision': prec, 'recall': rec, 'f1': f1})
                    
                    # TEDS
                    pred_html = pred_table.to_html(include_style=False)
                    gt_html = gt_table.get('html', '')
                    
                    if gt_html:
                        teds = calculate_teds(pred_html, gt_html)
                        all_teds.append(teds)
                    
                    # OCR metrics (per cell)
                    for pred_cell, gt_cell in zip(pred_cells, gt_cells):
                        pred_text = pred_cell.get('text', '')
                        gt_text = gt_cell.get('text', '')
                        
                        if gt_text:
                            all_cer.append(calculate_cer(pred_text, gt_text))
                            all_wer.append(calculate_wer(pred_text, gt_text))
                
                per_sample_results.append({
                    'sample_idx': idx,
                    'image_path': sample['image_path'],
                    'num_pred_tables': len(result.tables),
                    'num_gt_tables': len(gt_tables),
                    'detection': det_metrics,
                    'processing_time': result.total_time
                })
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                per_sample_results.append({
                    'sample_idx': idx,
                    'error': str(e)
                })
        
        # Aggregate metrics
        report = EvaluationReport(
            num_samples=len(dataset),
            per_sample_results=per_sample_results
        )
        
        if all_detection_metrics:
            report.detection_precision = np.mean([m['precision'] for m in all_detection_metrics])
            report.detection_recall = np.mean([m['recall'] for m in all_detection_metrics])
            report.detection_f1 = np.mean([m['f1'] for m in all_detection_metrics])
            report.detection_iou = np.mean([m['avg_iou'] for m in all_detection_metrics])
        
        if all_cell_f1:
            report.cell_precision = np.mean([m['precision'] for m in all_cell_f1])
            report.cell_recall = np.mean([m['recall'] for m in all_cell_f1])
            report.cell_f1 = np.mean([m['f1'] for m in all_cell_f1])
        
        if all_teds:
            report.teds_score = np.mean(all_teds)
        
        if all_cer:
            report.cer = np.mean(all_cer)
        
        if all_wer:
            report.wer = np.mean(all_wer)
        
        if all_times:
            report.avg_processing_time = np.mean(all_times)
        
        return report
    
    def evaluate_single(
        self,
        image_path: str,
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to image
            ground_truth: Ground truth annotations
            
        Returns:
            Dictionary with evaluation metrics
        """
        result = self.pipeline.extract(image_path)
        
        gt_tables = ground_truth.get('tables', [])
        
        # Detection metrics
        pred_boxes = [tuple(t.bbox) for t in result.tables if t.bbox]
        gt_boxes = [tuple(t['bbox']) for t in gt_tables if t.get('bbox')]
        
        det_metrics = calculate_detection_metrics(pred_boxes, gt_boxes)
        
        # Structure metrics
        cell_metrics = []
        teds_scores = []
        
        for pred_table, gt_table in zip(result.tables, gt_tables):
            pred_cells = [c.to_dict() for c in pred_table.cells]
            gt_cells = gt_table.get('cells', [])
            
            prec, rec, f1 = calculate_cell_f1(pred_cells, gt_cells)
            cell_metrics.append({'precision': prec, 'recall': rec, 'f1': f1})
            
            if gt_table.get('html'):
                teds = calculate_teds(
                    pred_table.to_html(include_style=False),
                    gt_table['html']
                )
                teds_scores.append(teds)
        
        return {
            'detection': det_metrics,
            'cell_f1': np.mean([m['f1'] for m in cell_metrics]) if cell_metrics else 0.0,
            'teds': np.mean(teds_scores) if teds_scores else 0.0,
            'processing_time': result.total_time,
            'num_tables': len(result.tables)
        }
    
    def load_dataset(
        self,
        annotations_path: Union[str, Path],
        images_dir: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Load evaluation dataset from COCO-style annotations.
        
        Args:
            annotations_path: Path to annotations JSON file
            images_dir: Directory containing images
            
        Returns:
            List of samples ready for evaluation
        """
        annotations_path = Path(annotations_path)
        images_dir = Path(images_dir)
        
        with open(annotations_path) as f:
            data = json.load(f)
        
        # Build image ID to file mapping
        images = {img['id']: img for img in data.get('images', [])}
        
        # Group annotations by image
        image_annotations = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Build dataset
        dataset = []
        for img_id, img_info in images.items():
            image_path = images_dir / img_info['file_name']
            
            if not image_path.exists():
                continue
            
            annotations = image_annotations.get(img_id, [])
            
            tables = []
            for ann in annotations:
                bbox = ann.get('bbox', [])
                if len(bbox) == 4:
                    # Convert [x, y, w, h] to [x1, y1, x2, y2]
                    x, y, w, h = bbox
                    bbox = [x, y, x + w, y + h]
                
                tables.append({
                    'bbox': bbox,
                    'cells': ann.get('cells', []),
                    'html': ann.get('html', ''),
                    'category_id': ann.get('category_id', 1)
                })
            
            dataset.append({
                'image_path': str(image_path),
                'tables': tables,
                'image_info': img_info
            })
        
        return dataset
    
    def save_report(
        self,
        report: EvaluationReport,
        output_path: Union[str, Path]
    ) -> None:
        """Save evaluation report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
