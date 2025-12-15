# Byte-sized Table Extraction System — Project Report

Date: 2025-12-15

## 1. Executive Summary
Byte-sized delivers an end-to-end, open-source system to detect tables in documents, recover their structure (rows, columns, merged cells), extract text (Korean + English), export to CSV/JSON/Excel/HTML, and provide a chatbot interface for semantic analysis over extracted tables.

## 2. Objectives
- Robust table detection and structure recognition across noisy, skewed, and complex layouts
- High-quality OCR for Korean/English documents
- Developer-friendly API and a professional single-page UI with Byte-sized branding
- Local-first deployment without Docker, optional GPU, and low-latency via model preloading
- Conversational insights via a chat interface over extracted tables

## 3. System Architecture
- API: FastAPI (`src/api`) with endpoints for extraction, export, health, feedback, stats, and chat
- Pipeline: `src/pipeline.py` orchestrates preprocessing → detection → structure → OCR → export
- Detection: Microsoft Table Transformer `microsoft/table-transformer-detection`
- Structure: Microsoft Table Transformer `microsoft/table-transformer-structure-recognition`
- OCR: EasyOCR (ko, en); optional PaddleOCR
- UI: `src/api/static/index.html` (enterprise-style, Byte-sized brand) with drag-drop upload, pipeline timeline, visualization, exports, and chatbot
- Caching: HF models cached under user cache; preloading on startup for fast requests

## 4. Key Features Implemented
- Table detection and structure recovery via HF Table Transformer models
- OCR with EasyOCR (Korean + English), cell-level text assignment
- Exports: CSV/JSON/HTML + multi-sheet Excel (one sheet per table)
- Chat endpoint `/api/v1/chat` performing lightweight retrieval over tables and optional LLM (Ollama) inference
- UI enhancements: Byte-sized brand header, visualization panels, export actions, chatbot panel in-page
- Startup warm-cache: preloads HF models and initializes OCR

## 5. Data Flow
1. Upload image/PDF → preprocessing (deskew/contrast configurable)
2. Table detection (HF detection model)
3. Structure extraction (HF structure model) → cells, merges
4. OCR (per cell; confidence stored)
5. Exports (CSV/JSON/HTML/Excel) + Chat context generation

## 6. Endpoints
- POST `/api/v1/extract` – run extraction, returns `job_id` and tables
- GET `/api/v1/export/{job_id}?format=csv|json|html|excel[&all_tables=true]`
- GET `/api/v1/health` – model + uptime status
- GET `/api/v1/stats` – simple counters
- POST `/api/v1/chat` – `{ job_id, question } → { answer, citations }`

## 7. Performance & Ops
- First-run latency minimized by `src/api/app.py` preloading models
- Subsequent runs hit local caches
- CPU-only works; GPU accelerates HF models and OCR

## 8. Evaluation Summary
- Metrics supported: Cell-F1, TEDS, IoU, CER/WER (see `src/evaluation`)
- For production, create a labeled set and run `TableExtractionEvaluator` to generate reports

## 9. Security & Privacy
- No cloud upload required; runs locally
- Authentication not enabled by default; add API keys / session auth for multi-user deployments

## 10. Limitations & Risks
- Extremely complex nested or borderless tables may require fine-tuning
- OCR errors propagate; consider dictionary-based post-correction for critical domains
- Chat quality depends on LLM availability (Ollama recommended)

## 11. Roadmap
- Add FAISS-based retriever for stronger chat grounding
- Pagination and multi-page PDF processing in UI
- Optional login/auth (JWT) and user workspaces
- Dataset curation and fine-tuning workflow

## 12. Installation & Run
```bash
# Create venv and install deps
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pre-download models (fast startup)
python scripts/download_models_fast.py

# Run API
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000 and use the UI.

## 13. Team & Credits
- Team: Byte-sized
- Open-source components: Microsoft Table Transformer, EasyOCR, FastAPI, Uvicorn, Pandas
