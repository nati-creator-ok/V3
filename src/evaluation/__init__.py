"""
Evaluation module for table extraction system.
Implements metrics: Cell-F1, TEDS, IoU, CER/WER.
"""

from .metrics import (
    calculate_cell_f1,
    calculate_teds,
    calculate_iou,
    calculate_cer,
    calculate_wer,
    EvaluationReport,
)
from .evaluator import TableExtractionEvaluator

__all__ = [
    "calculate_cell_f1",
    "calculate_teds",
    "calculate_iou",
    "calculate_cer",
    "calculate_wer",
    "EvaluationReport",
    "TableExtractionEvaluator",
]
