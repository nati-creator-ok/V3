# AI OCR-based Automatic Table Extraction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready AI system that automatically detects and extracts tabular data from document images/PDFs (simple grids, merged cells, nested tables, irregular layouts) and converts them into structured machine-readable outputs (CSV/JSON/Excel).

## 🌟 Features

- **Multi-format Support**: Process images (PNG, JPG, TIFF) and PDF documents
- **Table Detection**: YOLOv8-based detection for accurate table localization
- **Structure Recognition**: Transformer-based model for cell detection and span recognition
- **Multi-language OCR**: Korean, English, and mixed-language text extraction
- **Merged Cell Handling**: Automatic detection of rowspan/colspan
- **Export Formats**: CSV, JSON, Excel, HTML output
- **REST API**: FastAPI-based API for easy integration
- **Docker Support**: Production-ready containerized deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Development](#development)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- Tesseract OCR (optional, for Tesseract engine)

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/table-extraction-system.git
cd table-extraction-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (if available)
# python scripts/download_weights.py
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the API at http://localhost:8000
```

## 🏃 Quick Start

### Python API

```python
from src.pipeline import TableExtractionPipeline

# Initialize pipeline
pipeline = TableExtractionPipeline(device="cuda")

# Extract tables from image
result = pipeline.extract("document.png")

# Access extracted tables
for table in result.tables:
    print(f"Table with {table.num_rows} rows and {table.num_cols} columns")
    print(table.to_csv())
    
# Export to Excel
result.to_excel("output.xlsx")
```

### Command Line

```bash
# Extract tables from an image
python -m src.cli extract document.png --output results/

# Extract from PDF
python -m src.cli extract document.pdf --format csv --output results/

# Start API server
python -m src.cli serve --host 0.0.0.0 --port 8000
```

### REST API

```bash
# Extract tables from uploaded file
curl -X POST "http://localhost:8000/api/v1/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.png"

# Get result by job ID
curl "http://localhost:8000/api/v1/extract/{job_id}"

# Export to CSV
curl "http://localhost:8000/api/v1/export/{job_id}?format=csv" -o table.csv
```

## 📚 API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extract` | Upload and extract tables from document |
| `GET` | `/api/v1/extract/{job_id}` | Get extraction result |
| `POST` | `/api/v1/batch` | Batch extraction for multiple files |
| `GET` | `/api/v1/export/{job_id}` | Export result in specified format |
| `POST` | `/api/v1/feedback` | Submit correction feedback |
| `GET` | `/api/v1/health` | Health check |

### Response Format

```json
{
  "job_id": "uuid",
  "status": "completed",
  "tables": [
    {
      "table_id": "t1",
      "bbox": [x, y, width, height],
      "rows": 4,
      "cols": 3,
      "cells": [
        {
          "row": 0,
          "col": 0,
          "rowspan": 1,
          "colspan": 2,
          "text": "Invoice #123",
          "confidence": 0.98
        }
      ],
      "html": "<table>...</table>",
      "csv": "..."
    }
  ],
  "processing_time": 1.23
}
```

## 🏗️ Architecture

```
Input Document
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  ← Deskew, denoise, contrast enhancement
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Table Detection │  ← YOLOv8 / Faster R-CNN
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Structure     │  ← Transformer-based encoder-decoder
│  Recognition    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OCR Engine    │  ← EasyOCR / Tesseract (Korean + English)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │  ← Text normalization, cell assignment
└────────┬────────┘
         │
         ▼
    Structured Output (CSV/JSON/Excel/HTML)
```

### Module Overview

| Module | Description |
|--------|-------------|
| `src/preprocessing/` | Image preprocessing (deskew, denoise, binarization) |
| `src/detection/` | Table region detection using YOLO/Detectron2 |
| `src/structure/` | Table structure recognition (cells, rows, columns) |
| `src/ocr/` | OCR engines (EasyOCR, Tesseract, PaddleOCR) |
| `src/api/` | FastAPI REST API |
| `src/evaluation/` | Evaluation metrics (Cell-F1, TEDS, CER/WER) |

## ⚙️ Configuration

### Environment Variables

```bash
# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Model Settings
MODEL_DEVICE=cuda  # cuda, cpu, mps
TABLE_DETECTION_MODEL=yolov8
OCR_ENGINE=easyocr
OCR_LANGUAGES=ko,en

# Processing
MAX_IMAGE_SIZE=4096
CONFIDENCE_THRESHOLD=0.5

# Database
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379/0
```

See `.env.example` for all configuration options.

## 📊 Evaluation

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Cell-F1 | Precision/recall on cell detection | ≥90% simple, ≥80% complex |
| TEDS | Tree-Edit-Distance Similarity | ≥0.9 |
| IoU | Intersection over Union for detection | ≥0.8 |
| CER | Character Error Rate | ≤5% |
| WER | Word Error Rate | ≤10% |

### Running Evaluation

```python
from src.evaluation import TableExtractionEvaluator
from src.pipeline import TableExtractionPipeline

pipeline = TableExtractionPipeline()
evaluator = TableExtractionEvaluator(pipeline)

# Load test dataset
dataset = evaluator.load_dataset("annotations.json", "images/")

# Run evaluation
report = evaluator.evaluate(dataset)
print(report)

# Save report
evaluator.save_report(report, "evaluation_report.json")
```

## 🐳 Deployment

### Docker Compose

```bash
# Production deployment
docker-compose up -d

# With GPU support
docker-compose --profile gpu up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### Kubernetes (Optional)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

### Scaling

- Use `docker-compose scale api=3` for horizontal scaling
- Configure nginx/traefik as load balancer
- Use Redis for session storage and caching

## 🔧 Development

### Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
V3/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Main extraction pipeline
│   ├── cli.py              # Command-line interface
│   ├── preprocessing/      # Image preprocessing
│   ├── detection/          # Table detection
│   ├── structure/          # Structure recognition
│   ├── ocr/                # OCR engines
│   ├── api/                # REST API
│   └── evaluation/         # Evaluation metrics
├── models/
│   └── weights/            # Model weights
├── data/
│   ├── uploads/            # Uploaded files
│   └── exports/            # Exported results
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🗓️ Roadmap

- [ ] Week 1-3: Data collection and preprocessing pipeline
- [ ] Week 3-6: Table detection and baseline OCR
- [ ] Week 6-8: Structure recognition model
- [ ] Week 8-10: Optimization and evaluation
- [ ] Week 10-12: Web UI, API, and deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet)
- [TableFormer](https://github.com/microsoft/table-transformer)

## 📧 Contact

For questions or support, please open an issue on GitHub.
