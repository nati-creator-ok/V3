"""
Tests for evaluation metrics.
"""

import pytest
from src.evaluation.metrics import (
    calculate_iou,
    calculate_cell_f1,
    calculate_teds,
    calculate_cer,
    calculate_wer,
    calculate_detection_metrics,
)


class TestIoU:
    """Test IoU calculation."""
    
    def test_perfect_overlap(self):
        box = (0, 0, 100, 100)
        assert calculate_iou(box, box) == 1.0
    
    def test_no_overlap(self):
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        assert calculate_iou(box1, box2) == 0.0
    
    def test_partial_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert abs(iou - expected) < 0.01
    
    def test_contained_box(self):
        box1 = (0, 0, 100, 100)
        box2 = (25, 25, 75, 75)
        iou = calculate_iou(box1, box2)
        # Intersection = box2 area = 2500
        # Union = box1 area = 10000
        expected = 2500 / 10000
        assert abs(iou - expected) < 0.01


class TestCellF1:
    """Test cell-level F1 calculation."""
    
    def test_perfect_match(self):
        cells = [
            {"row": 0, "col": 0, "text": "A"},
            {"row": 0, "col": 1, "text": "B"},
            {"row": 1, "col": 0, "text": "C"},
        ]
        prec, rec, f1 = calculate_cell_f1(cells, cells)
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0
    
    def test_no_match(self):
        pred = [{"row": 0, "col": 0, "text": "A"}]
        gt = [{"row": 1, "col": 1, "text": "B"}]
        prec, rec, f1 = calculate_cell_f1(pred, gt)
        assert f1 == 0.0
    
    def test_partial_match(self):
        pred = [
            {"row": 0, "col": 0, "text": "A"},
            {"row": 0, "col": 1, "text": "B"},
        ]
        gt = [
            {"row": 0, "col": 0, "text": "A"},
            {"row": 0, "col": 1, "text": "X"},  # Different text
        ]
        prec, rec, f1 = calculate_cell_f1(pred, gt, text_match=True)
        assert prec == 0.5
        assert rec == 0.5
        assert f1 == 0.5
    
    def test_without_text_match(self):
        pred = [{"row": 0, "col": 0, "text": "A"}]
        gt = [{"row": 0, "col": 0, "text": "B"}]
        prec, rec, f1 = calculate_cell_f1(pred, gt, text_match=False)
        assert f1 == 1.0
    
    def test_empty_predictions(self):
        pred = []
        gt = [{"row": 0, "col": 0, "text": "A"}]
        prec, rec, f1 = calculate_cell_f1(pred, gt)
        assert f1 == 0.0
    
    def test_empty_ground_truth(self):
        pred = [{"row": 0, "col": 0, "text": "A"}]
        gt = []
        prec, rec, f1 = calculate_cell_f1(pred, gt)
        assert prec == 0.0
        # When no ground truth, recall is undefined or 0


class TestOCRMetrics:
    """Test OCR metrics (CER, WER)."""
    
    def test_cer_perfect(self):
        assert calculate_cer("hello", "hello") == 0.0
    
    def test_cer_one_char_error(self):
        cer = calculate_cer("hello", "hallo")  # 1 substitution
        assert abs(cer - 0.2) < 0.01  # 1/5 = 0.2
    
    def test_cer_empty_prediction(self):
        cer = calculate_cer("", "hello")
        assert cer == 1.0
    
    def test_cer_empty_ground_truth(self):
        # Empty ground truth - no reference to compare, CER is 0 (or undefined)
        cer = calculate_cer("hello", "")
        # Implementation may return 0 or 1 depending on definition
        assert cer in [0.0, 1.0]
    
    def test_wer_perfect(self):
        assert calculate_wer("hello world", "hello world") == 0.0
    
    def test_wer_one_word_error(self):
        wer = calculate_wer("hello world", "hello earth")
        assert abs(wer - 0.5) < 0.01  # 1/2 = 0.5
    
    def test_wer_extra_word(self):
        wer = calculate_wer("hello world today", "hello world")
        assert wer > 0


class TestTEDS:
    """Test TEDS calculation."""
    
    def test_identical_html(self):
        html = "<table><tr><td>A</td></tr></table>"
        score = calculate_teds(html, html)
        assert score == 1.0
    
    def test_different_html(self):
        html1 = "<table><tr><td>A</td></tr></table>"
        html2 = "<table><tr><td>B</td></tr></table>"
        score = calculate_teds(html1, html2)
        assert score < 1.0
    
    def test_structure_only(self):
        html1 = "<table><tr><td>A</td></tr></table>"
        html2 = "<table><tr><td>B</td></tr></table>"
        score = calculate_teds(html1, html2, structure_only=True)
        assert score == 1.0  # Structure is the same


class TestDetectionMetrics:
    """Test detection metrics."""
    
    def test_perfect_detection(self):
        boxes = [(0, 0, 100, 100)]
        metrics = calculate_detection_metrics(boxes, boxes)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_no_detections(self):
        pred = []
        gt = [(0, 0, 100, 100)]
        metrics = calculate_detection_metrics(pred, gt)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
    
    def test_false_positive(self):
        pred = [(0, 0, 100, 100), (200, 200, 300, 300)]
        gt = [(0, 0, 100, 100)]
        metrics = calculate_detection_metrics(pred, gt)
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0
