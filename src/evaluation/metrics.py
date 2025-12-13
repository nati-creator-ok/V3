"""
Evaluation metrics for table extraction.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    
    # Detection metrics
    detection_precision: float = 0.0
    detection_recall: float = 0.0
    detection_f1: float = 0.0
    detection_iou: float = 0.0
    
    # Structure metrics
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    teds_score: float = 0.0
    
    # OCR metrics
    cer: float = 0.0
    wer: float = 0.0
    
    # Timing
    avg_processing_time: float = 0.0
    
    # Details
    num_samples: int = 0
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection": {
                "precision": self.detection_precision,
                "recall": self.detection_recall,
                "f1": self.detection_f1,
                "iou": self.detection_iou
            },
            "structure": {
                "cell_precision": self.cell_precision,
                "cell_recall": self.cell_recall,
                "cell_f1": self.cell_f1,
                "teds": self.teds_score
            },
            "ocr": {
                "cer": self.cer,
                "wer": self.wer
            },
            "performance": {
                "avg_processing_time": self.avg_processing_time,
                "num_samples": self.num_samples
            }
        }
    
    def __str__(self) -> str:
        return f"""
Evaluation Report ({self.num_samples} samples)
================================================
Detection:
  Precision: {self.detection_precision:.4f}
  Recall:    {self.detection_recall:.4f}
  F1:        {self.detection_f1:.4f}
  IoU:       {self.detection_iou:.4f}

Structure:
  Cell Precision: {self.cell_precision:.4f}
  Cell Recall:    {self.cell_recall:.4f}
  Cell F1:        {self.cell_f1:.4f}
  TEDS:           {self.teds_score:.4f}

OCR:
  CER: {self.cer:.4f}
  WER: {self.wer:.4f}

Performance:
  Avg Processing Time: {self.avg_processing_time:.3f}s
"""


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)
        
    Returns:
        IoU score (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_cell_f1(
    pred_cells: List[Dict],
    gt_cells: List[Dict],
    iou_threshold: float = 0.5,
    text_match: bool = True
) -> Tuple[float, float, float]:
    """
    Calculate cell-level F1 score.
    
    Args:
        pred_cells: List of predicted cells with 'row', 'col', 'text', 'bbox'
        gt_cells: List of ground truth cells
        iou_threshold: IoU threshold for matching (if using bbox)
        text_match: Whether to require text match
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not gt_cells:
        return (1.0, 1.0, 1.0) if not pred_cells else (0.0, 0.0, 0.0)
    
    if not pred_cells:
        return (0.0, 0.0, 0.0)
    
    matched_gt = set()
    matched_pred = set()
    
    for p_idx, pred in enumerate(pred_cells):
        for g_idx, gt in enumerate(gt_cells):
            if g_idx in matched_gt:
                continue
            
            # Check position match
            position_match = (
                pred.get('row') == gt.get('row') and
                pred.get('col') == gt.get('col')
            )
            
            # Check bbox IoU if available
            if pred.get('bbox') and gt.get('bbox') and not position_match:
                iou = calculate_iou(
                    tuple(pred['bbox']),
                    tuple(gt['bbox'])
                )
                position_match = iou >= iou_threshold
            
            if not position_match:
                continue
            
            # Check text match if required
            if text_match:
                pred_text = str(pred.get('text', '')).strip().lower()
                gt_text = str(gt.get('text', '')).strip().lower()
                
                if pred_text != gt_text:
                    continue
            
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            break
    
    tp = len(matched_gt)
    fp = len(pred_cells) - len(matched_pred)
    fn = len(gt_cells) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def _edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1]   # replace
                )
    
    return dp[m][n]


def calculate_teds(
    pred_html: str,
    gt_html: str,
    structure_only: bool = False
) -> float:
    """
    Calculate Tree Edit Distance-based Similarity (TEDS) score.
    
    TEDS measures structural similarity between predicted and ground truth tables.
    
    Args:
        pred_html: Predicted table as HTML string
        gt_html: Ground truth table as HTML string
        structure_only: If True, ignore text content
        
    Returns:
        TEDS score (0-1, higher is better)
    """
    try:
        from lxml import html as lxml_html
    except ImportError:
        # Fallback to simple string comparison
        if structure_only:
            # Remove text content
            pred_html = re.sub(r'>([^<]+)<', '><', pred_html)
            gt_html = re.sub(r'>([^<]+)<', '><', gt_html)
        
        # Normalize whitespace
        pred_html = re.sub(r'\s+', '', pred_html)
        gt_html = re.sub(r'\s+', '', gt_html)
        
        if pred_html == gt_html:
            return 1.0
        
        # Use edit distance as approximation
        edit_dist = _edit_distance(pred_html, gt_html)
        max_len = max(len(pred_html), len(gt_html))
        
        return 1.0 - (edit_dist / max_len) if max_len > 0 else 1.0
    
    # Parse HTML trees
    try:
        pred_tree = lxml_html.fromstring(pred_html)
        gt_tree = lxml_html.fromstring(gt_html)
    except Exception:
        return 0.0
    
    def tree_edit_distance(node1, node2):
        """Calculate tree edit distance recursively."""
        if node1 is None and node2 is None:
            return 0
        if node1 is None:
            return _count_nodes(node2)
        if node2 is None:
            return _count_nodes(node1)
        
        # Compare nodes
        cost = 0
        if node1.tag != node2.tag:
            cost += 1
        
        if not structure_only:
            text1 = (node1.text or '').strip()
            text2 = (node2.text or '').strip()
            if text1 != text2:
                cost += _edit_distance(text1, text2) / max(len(text1), len(text2), 1)
        
        # Compare children
        children1 = list(node1)
        children2 = list(node2)
        
        m, n = len(children1), len(children2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = sum(_count_nodes(children1[j]) for j in range(i))
        for j in range(n + 1):
            dp[0][j] = sum(_count_nodes(children2[k]) for k in range(j))
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + _count_nodes(children1[i - 1]),
                    dp[i][j - 1] + _count_nodes(children2[j - 1]),
                    dp[i - 1][j - 1] + tree_edit_distance(children1[i - 1], children2[j - 1])
                )
        
        return cost + dp[m][n]
    
    def _count_nodes(node):
        """Count nodes in tree."""
        if node is None:
            return 0
        return 1 + sum(_count_nodes(child) for child in node)
    
    # Calculate TEDS
    edit_dist = tree_edit_distance(pred_tree, gt_tree)
    max_nodes = max(_count_nodes(pred_tree), _count_nodes(gt_tree))
    
    return 1.0 - (edit_dist / max_nodes) if max_nodes > 0 else 1.0


def calculate_cer(pred_text: str, gt_text: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
        
    Returns:
        CER (0 = perfect, higher = worse)
    """
    pred_text = pred_text.strip()
    gt_text = gt_text.strip()
    
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    
    edit_dist = _edit_distance(pred_text, gt_text)
    return edit_dist / len(gt_text)


def calculate_wer(pred_text: str, gt_text: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        pred_text: Predicted text
        gt_text: Ground truth text
        
    Returns:
        WER (0 = perfect, higher = worse)
    """
    pred_words = pred_text.strip().split()
    gt_words = gt_text.strip().split()
    
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    
    # Build word-level edit distance
    m, n = len(pred_words), len(gt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == gt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )
    
    return dp[m][n] / len(gt_words)


def calculate_detection_metrics(
    pred_boxes: List[Tuple[float, float, float, float]],
    gt_boxes: List[Tuple[float, float, float, float]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate detection metrics (precision, recall, F1, mAP).
    
    Args:
        pred_boxes: Predicted bounding boxes
        gt_boxes: Ground truth bounding boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with precision, recall, f1, avg_iou
    """
    if not gt_boxes:
        return {
            "precision": 1.0 if not pred_boxes else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_boxes else 0.0,
            "avg_iou": 1.0
        }
    
    if not pred_boxes:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_iou": 0.0
        }
    
    # Match predictions to ground truth
    matched_gt = set()
    matched_ious = []
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx, gt_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            matched_ious.append(best_iou)
    
    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_iou": avg_iou
    }
