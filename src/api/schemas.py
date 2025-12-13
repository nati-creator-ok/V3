"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of extraction job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportFormat(str, Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"


# Request Models

class ExtractionRequest(BaseModel):
    """Request for table extraction."""
    skip_preprocessing: bool = Field(default=False, description="Skip image preprocessing")
    skip_ocr: bool = Field(default=False, description="Skip OCR (structure only)")
    languages: List[str] = Field(default=["ko", "en"], description="OCR languages")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1, description="Confidence threshold")
    export_format: ExportFormat = Field(default=ExportFormat.JSON, description="Output format")


class BatchExtractionRequest(BaseModel):
    """Request for batch extraction."""
    options: ExtractionRequest = Field(default_factory=ExtractionRequest)


class FeedbackRequest(BaseModel):
    """User feedback for corrections."""
    job_id: str = Field(..., description="Job ID")
    table_index: int = Field(default=0, description="Table index in results")
    corrections: Dict[str, Any] = Field(..., description="Correction data")
    comment: Optional[str] = Field(default=None, description="User comment")


# Response Models

class CellResponse(BaseModel):
    """Response for a single cell."""
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    text: str = ""
    confidence: float = 1.0
    bbox: Optional[List[float]] = None
    cell_type: str = "data"


class TableResponse(BaseModel):
    """Response for a single table."""
    table_id: str
    bbox: Optional[List[float]] = None
    rows: int
    cols: int
    cells: List[CellResponse]
    confidence: float = 1.0
    html: Optional[str] = None
    csv: Optional[str] = None


class ExtractionResponse(BaseModel):
    """Response for extraction result."""
    job_id: str
    status: JobStatus
    tables: List[TableResponse] = Field(default_factory=list)
    num_tables: int = 0
    processing_time: float = 0.0
    page_number: int = 0
    source_file: Optional[str] = None
    image_size: List[int] = Field(default_factory=lambda: [0, 0])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction."""
    batch_id: str
    status: JobStatus
    total_files: int
    processed_files: int
    results: List[ExtractionResponse] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    feedback_id: str
    job_id: str
    status: str = "received"
    message: str = "Feedback recorded successfully"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
