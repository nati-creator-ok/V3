# API Documentation

This document provides detailed information about the Table Extraction API endpoints.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

API authentication can be enabled via API keys. Include the key in the request header:

```
Authorization: Bearer <your-api-key>
```

## Endpoints

### Health Check

Check API health status.

```
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Extract Tables

Extract tables from a single document image or PDF.

```
POST /api/v1/extract
```

**Request:**
- `Content-Type`: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | Image or PDF file |
| page | int | No | Page number for PDF (default: all) |
| output_format | string | No | Output format: `json`, `csv`, `excel` |
| detect_merged | bool | No | Detect merged cells (default: true) |
| ocr_language | string | No | OCR language code: `en`, `ko`, `en+ko` |

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/extract" \
     -H "accept: application/json" \
     -F "file=@document.png" \
     -F "output_format=json" \
     -F "ocr_language=en+ko"
```

**Response:**
```json
{
    "success": true,
    "document_id": "doc_abc123",
    "tables": [
        {
            "table_id": 0,
            "bbox": [100, 200, 800, 600],
            "confidence": 0.95,
            "num_rows": 5,
            "num_cols": 4,
            "cells": [
                {
                    "row": 0,
                    "col": 0,
                    "row_span": 1,
                    "col_span": 1,
                    "text": "Header",
                    "bbox": [100, 200, 275, 280],
                    "confidence": 0.92
                }
            ],
            "html": "<table>...</table>",
            "data": [
                ["Header", "Col2", "Col3", "Col4"],
                ["A", "B", "C", "D"]
            ]
        }
    ],
    "processing_time": 1.234,
    "metadata": {
        "image_size": [2480, 3508],
        "num_tables_detected": 1
    }
}
```

**Error Response:**
```json
{
    "success": false,
    "error": "Unsupported file format",
    "error_code": "INVALID_FORMAT",
    "details": "File must be PNG, JPEG, or PDF"
}
```

---

### Batch Extract

Extract tables from multiple documents.

```
POST /api/v1/batch
```

**Request:**
- `Content-Type`: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| files | file[] | Yes | Multiple image/PDF files |
| output_format | string | No | Output format |
| async | bool | No | Async processing (default: false) |

**Response:**
```json
{
    "success": true,
    "batch_id": "batch_xyz789",
    "results": [
        {
            "filename": "doc1.png",
            "document_id": "doc_abc123",
            "tables_count": 2,
            "status": "completed"
        },
        {
            "filename": "doc2.pdf",
            "document_id": "doc_def456",
            "tables_count": 1,
            "status": "completed"
        }
    ],
    "total_processing_time": 3.456
}
```

---

### Export Table

Export extracted table in specific format.

```
POST /api/v1/export
```

**Request Body:**
```json
{
    "document_id": "doc_abc123",
    "table_id": 0,
    "format": "excel",
    "options": {
        "include_header": true,
        "merge_cells": true,
        "sheet_name": "Extracted Table"
    }
}
```

**Response:**
- Returns file download with appropriate content type

---

### Get Processing Status

Check status of async batch processing.

```
GET /api/v1/status/{batch_id}
```

**Response:**
```json
{
    "batch_id": "batch_xyz789",
    "status": "processing",
    "progress": 0.65,
    "completed": 13,
    "total": 20,
    "estimated_remaining": 45.5
}
```

---

### Submit Feedback

Submit feedback for extracted results (active learning).

```
POST /api/v1/feedback
```

**Request Body:**
```json
{
    "document_id": "doc_abc123",
    "table_id": 0,
    "rating": 4,
    "corrections": {
        "cells": [
            {
                "row": 1,
                "col": 2,
                "corrected_text": "Actual text"
            }
        ]
    },
    "comments": "Merged cell not detected correctly"
}
```

**Response:**
```json
{
    "success": true,
    "feedback_id": "fb_123456",
    "message": "Feedback recorded successfully"
}
```

---

### Model Information

Get information about loaded models.

```
GET /api/v1/models/info
```

**Response:**
```json
{
    "detection_model": {
        "name": "YOLOv8 Table Detector",
        "version": "1.0.0",
        "loaded": true,
        "device": "cuda:0"
    },
    "structure_model": {
        "name": "Table Transformer",
        "version": "1.0.0",
        "loaded": true,
        "device": "cuda:0"
    },
    "ocr_engine": {
        "name": "EasyOCR",
        "languages": ["en", "ko"],
        "loaded": true
    }
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_FORMAT | Unsupported file format |
| FILE_TOO_LARGE | File exceeds size limit |
| PROCESSING_ERROR | Error during extraction |
| MODEL_NOT_LOADED | Required model not available |
| RATE_LIMITED | Too many requests |
| INVALID_PARAMS | Invalid request parameters |

## Rate Limiting

- Default: 100 requests per minute per IP
- Configurable via environment variables

## WebSocket API

For real-time processing updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/extract');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'extract',
        document_id: 'doc_abc123'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Progress:', data.progress);
};
```

## SDK Examples

### Python

```python
import requests

# Single file extraction
with open('document.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/extract',
        files={'file': f},
        data={'output_format': 'json'}
    )

result = response.json()
print(f"Found {len(result['tables'])} tables")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('output_format', 'json');

const response = await fetch('http://localhost:8000/api/v1/extract', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Found ${result.tables.length} tables`);
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```
GET /openapi.json
GET /docs  (Swagger UI)
GET /redoc (ReDoc)
```
