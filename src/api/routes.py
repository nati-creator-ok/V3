"""
API route handlers for table extraction endpoints.
"""

import io
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse

from src import __version__
from src.config import settings

from .app import get_pipeline, get_uptime
from .schemas import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    CellResponse,
    ErrorResponse,
    ExportFormat,
    ExtractionRequest,
    ExtractionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    JobStatus,
    TableResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extraction"])

# In-memory job storage (use Redis/DB in production)
_jobs = {}
_feedback = []


class ChatRequest(BaseModel):
    job_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict] = []


def _convert_extraction_result(result, job_id: str) -> ExtractionResponse:
    """Convert pipeline result to API response."""
    tables = []
    
    for idx, table in enumerate(result.tables):
        cells = [
            CellResponse(
                row=cell.row,
                col=cell.col,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                text=cell.text,
                confidence=cell.confidence,
                bbox=list(cell.bbox) if cell.bbox else None,
                cell_type=cell.cell_type.value
            )
            for cell in table.cells
        ]
        
        tables.append(TableResponse(
            table_id=f"t{idx + 1}",
            bbox=list(table.bbox) if table.bbox else None,
            rows=table.num_rows,
            cols=table.num_cols,
            cells=cells,
            confidence=table.confidence,
            html=table.to_html(),
            csv=table.to_csv()
        ))
    
    return ExtractionResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        tables=tables,
        num_tables=len(tables),
        processing_time=result.total_time,
        page_number=result.page_number,
        source_file=result.source_file,
        image_size=list(result.image_size)
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and model status.
    """
    pipeline = get_pipeline()
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded={
            "detector": (pipeline.detector.model is not None) if pipeline else False,
            "structure": bool(getattr(pipeline, "structure_extractor", None)),
            "ocr": pipeline.ocr_engine.is_initialized if pipeline else False
        },
        uptime_seconds=get_uptime()
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_over_tables(req: ChatRequest):
    """Chat over extracted tables using naive retrieval; calls Ollama if available."""
    if req.job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[req.job_id]

    # Build text chunks from tables
    chunks = []
    for i, tbl in enumerate(job.tables):
        # group cell texts by row
        rows = [[] for _ in range(tbl.rows)]
        for cell in tbl.cells:
            if 0 <= cell.row < tbl.rows:
                rows[cell.row].append(cell.text or "")
        row_texts = [" | ".join(r) for r in rows]
        chunk_text = f"Table {i+1} ({tbl.rows}x{tbl.cols})\n" + "\n".join(row_texts)
        chunks.append({"table_index": i, "text": chunk_text})

    # Naive retrieval: rank by keyword overlap
    q_tokens = set([t.lower() for t in req.question.split() if len(t) > 2])
    def score(text: str) -> int:
        return sum(1 for t in q_tokens if t in text.lower())
    ranked = sorted(chunks, key=lambda c: score(c["text"]), reverse=True)[:5]

    context = "\n\n".join([f"[Table {c['table_index']+1}]\n{c['text']}" for c in ranked])
    prompt = (
        "Answer the question using the tables. "
        "Cite table indices you used. If arithmetic is needed, show steps briefly.\n\n"
        f"Tables:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    answer = ""
    citations = [{"table_index": c["table_index"]} for c in ranked]
    # Try Ollama (optional)
    try:
        import requests
        r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.1", "prompt": prompt, "stream": False}, timeout=10)
        if r.ok:
            data = r.json()
            answer = data.get("response", "")
        else:
            answer = "(LLM unavailable) " + prompt[:400]
    except Exception:
        answer = "(LLM unavailable) " + prompt[:400]

    return ChatResponse(answer=answer, citations=citations)


@router.post("/extract", response_model=ExtractionResponse)
async def extract_table(
    file: UploadFile = File(..., description="Image or PDF file"),
    skip_preprocessing: bool = Query(False, description="Skip preprocessing (not recommended)"),
    skip_ocr: bool = Query(False, description="Skip OCR"),
    languages: str = Query("ko,en", description="OCR languages (comma-separated)"),
    confidence_threshold: float = Query(0.5, ge=0, le=1, description="Confidence threshold")
):
    """
    Extract tables from uploaded document.
    
    Supports image formats (PNG, JPG, TIFF) and PDF files.
    Returns detected tables with structure and text content.
    """
    job_id = str(uuid.uuid4())
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Determine file type
        filename = file.filename or "upload"
        suffix = Path(filename).suffix.lower()
        
        pipeline = get_pipeline()
        
        if suffix == ".pdf":
            # Save PDF temporarily and process
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            
            try:
                results = pipeline.extract_from_pdf(tmp_path)
                # Return first page for single file upload
                result = results[0] if results else None
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # Process as image
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            result = pipeline.extract(
                image,
                skip_preprocessing=skip_preprocessing,
                skip_ocr=skip_ocr
            )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Extraction failed")
        
        response = _convert_extraction_result(result, job_id)
        response.source_file = filename
        
        # Store job result
        _jobs[job_id] = response
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extract/{job_id}", response_model=ExtractionResponse)
async def get_extraction_result(job_id: str):
    """
    Get extraction result by job ID.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return _jobs[job_id]


@router.post("/batch", response_model=BatchExtractionResponse)
async def batch_extract(
    files: List[UploadFile] = File(..., description="Multiple files"),
    options: BatchExtractionRequest = None
):
    """
    Extract tables from multiple files.
    """
    batch_id = str(uuid.uuid4())
    results = []
    
    pipeline = get_pipeline()
    
    for file in files:
        job_id = str(uuid.uuid4())
        
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                result = pipeline.extract(image)
                response = _convert_extraction_result(result, job_id)
                response.source_file = file.filename
                results.append(response)
            else:
                results.append(ExtractionResponse(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=f"Invalid image: {file.filename}"
                ))
        except Exception as e:
            results.append(ExtractionResponse(
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            ))
    
    return BatchExtractionResponse(
        batch_id=batch_id,
        status=JobStatus.COMPLETED,
        total_files=len(files),
        processed_files=len(results),
        results=results
    )


@router.get("/export/{job_id}")
async def export_result(
    job_id: str,
    format: ExportFormat = Query(ExportFormat.CSV, description="Export format"),
    table_index: int = Query(0, ge=0, description="Table index to export"),
    all_tables: bool = Query(False, description="Export all tables together")
):
    """
    Export extraction result in specified format.
    Supports exporting single table or all tables.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    if not all_tables:
        # Export single table
        if table_index >= len(job.tables):
            raise HTTPException(status_code=400, detail="Table index out of range")
        table = job.tables[table_index]
    else:
        # Export all tables
        table = None
    
    if format == ExportFormat.CSV:
        if all_tables:
            # Combine all tables with separators
            combined_csv = ""
            for i, tbl in enumerate(job.tables):
                combined_csv += f"# Table {i+1}\n{tbl.csv}\n\n"
            return StreamingResponse(
                io.StringIO(combined_csv),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=all_tables.csv"}
            )
        else:
            return StreamingResponse(
                io.StringIO(table.csv),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=table_{table_index}.csv"}
            )
    
    elif format == ExportFormat.HTML:
        if all_tables:
            # Combine all tables in HTML
            combined_html = "<html><body>\n"
            for i, tbl in enumerate(job.tables):
                combined_html += f"<h2>Table {i+1}</h2>\n{tbl.html}\n<br>\n"
            combined_html += "</body></html>"
            return StreamingResponse(
                io.StringIO(combined_html),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=all_tables.html"}
            )
        else:
            return StreamingResponse(
                io.StringIO(table.html),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=table_{table_index}.html"}
            )
    
    elif format == ExportFormat.JSON:
        import json
        if all_tables:
            # Export all tables as JSON array
            all_data = [tbl.model_dump() for tbl in job.tables]
            return StreamingResponse(
                io.StringIO(json.dumps(all_data, indent=2)),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=all_tables.json"}
            )
        else:
            return StreamingResponse(
                io.StringIO(json.dumps(table.model_dump(), indent=2)),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=table_{table_index}.json"}
            )
    
    elif format == ExportFormat.EXCEL:
        # Create Excel file
        import pandas as pd
        
        output = io.BytesIO()
        
        if all_tables:
            # Export each table to separate sheet
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for i, tbl in enumerate(job.tables):
                    grid = [["" for _ in range(tbl.cols)] for _ in range(tbl.rows)]
                    for cell in tbl.cells:
                        if cell.row < tbl.rows and cell.col < tbl.cols:
                            grid[cell.row][cell.col] = cell.text
                    df = pd.DataFrame(grid)
                    df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False, header=False)
        else:
            # Export single table
            grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
            for cell in table.cells:
                if cell.row < table.rows and cell.col < table.cols:
                    grid[cell.row][cell.col] = cell.text
            
            df = pd.DataFrame(grid)
            df.to_excel(output, index=False, header=False)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=table_{table_index}.xlsx"}
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit correction feedback for extracted tables.
    
    Feedback is used to improve model accuracy over time.
    """
    feedback_id = str(uuid.uuid4())
    
    _feedback.append({
        "feedback_id": feedback_id,
        "job_id": feedback.job_id,
        "table_index": feedback.table_index,
        "corrections": feedback.corrections,
        "comment": feedback.comment,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    logger.info(f"Feedback received for job {feedback.job_id}: {feedback.corrections}")
    
    return FeedbackResponse(
        feedback_id=feedback_id,
        job_id=feedback.job_id,
        status="received",
        message="Feedback recorded successfully"
    )


@router.get("/stats")
async def get_stats():
    """
    Get API usage statistics.
    """
    return {
        "total_jobs": len(_jobs),
        "total_feedback": len(_feedback),
        "uptime_seconds": get_uptime()
    }
