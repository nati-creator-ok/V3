# Annotation Format Schema

This document describes the annotation format used for training table detection and structure recognition models.

## Detection Annotations

Detection annotations are stored in COCO format for compatibility with most detection frameworks.

### COCO Format Structure

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "document_001.png",
            "width": 2480,
            "height": 3508
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, width, height],
            "area": 150000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {"id": 1, "name": "table", "supercategory": "document_element"},
        {"id": 2, "name": "table_rotated", "supercategory": "document_element"}
    ]
}
```

### YOLO Format

For YOLO training, annotations are converted to YOLO format:

```
# Each line: class_id center_x center_y width height (normalized 0-1)
0 0.5 0.5 0.3 0.2
```

## Structure Recognition Annotations

Structure annotations describe the cell-level structure of detected tables.

### Table Structure JSON Format

```json
{
    "image_id": "document_001_table_1",
    "table_bbox": [100, 200, 800, 600],
    "num_rows": 5,
    "num_cols": 4,
    "has_header": true,
    "cells": [
        {
            "id": 0,
            "row": 0,
            "col": 0,
            "row_span": 1,
            "col_span": 2,
            "bbox": [100, 200, 400, 250],
            "text": "Header spanning 2 columns",
            "is_header": true,
            "is_merged": true
        },
        {
            "id": 1,
            "row": 0,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "bbox": [400, 200, 600, 250],
            "text": "Column 3",
            "is_header": true,
            "is_merged": false
        }
    ],
    "row_separators": [200, 250, 350, 450, 550, 600],
    "col_separators": [100, 400, 600, 700, 800]
}
```

### HTML Ground Truth

For TEDS evaluation, ground truth HTML is also stored:

```html
<table>
    <thead>
        <tr>
            <th colspan="2">Header spanning 2 columns</th>
            <th>Column 3</th>
            <th>Column 4</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Row 1, Col 1</td>
            <td>Row 1, Col 2</td>
            <td>Row 1, Col 3</td>
            <td>Row 1, Col 4</td>
        </tr>
    </tbody>
</table>
```

## OCR Ground Truth

OCR ground truth for text recognition evaluation:

```json
{
    "image_id": "document_001",
    "text_regions": [
        {
            "id": 0,
            "bbox": [100, 200, 300, 250],
            "text": "Sample text content",
            "language": "en",
            "confidence": 1.0
        }
    ],
    "full_text": "Complete document text...",
    "language": "en"
}
```

## Dataset Organization

```
data/
├── raw/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── pdfs/
├── processed/
│   ├── images/
│   └── crops/
├── annotations/
│   ├── detection/
│   │   ├── train_coco.json
│   │   ├── val_coco.json
│   │   └── test_coco.json
│   ├── structure/
│   │   ├── train_structure.json
│   │   ├── val_structure.json
│   │   └── test_structure.json
│   └── ocr/
│       ├── train_ocr.json
│       └── val_ocr.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Supported Datasets

### PubTables-1M
- 947,642 tables for detection
- 758,849 tables with structure annotations
- Source: https://github.com/microsoft/table-transformer

### TableBank
- 417,000+ document images with table regions
- Source: https://doc-analysis.github.io/tablebank-page/

### SciTSR
- 15,000+ table images with structure annotations
- Focus on scientific documents

### FinTabNet
- 100,000+ tables from financial documents
- Complex layouts with merged cells

### Custom Dataset Preparation

1. **Image Collection**: Gather document images/PDFs
2. **Table Detection**: Annotate table bounding boxes using tools like LabelImg
3. **Structure Annotation**: Use custom tool or manual annotation for cells
4. **OCR Annotation**: Can be auto-generated or manually corrected
5. **Split Creation**: Create train/val/test splits (typically 80/10/10)

## Annotation Tools

- **LabelImg**: Bounding box annotation
- **VIA (VGG Image Annotator)**: Polygon and region annotation
- **CVAT**: Advanced annotation with custom attributes
- **Custom Tool**: See `scripts/annotate_tables.py`
