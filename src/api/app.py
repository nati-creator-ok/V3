"""
FastAPI application factory and configuration.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings

logger = logging.getLogger(__name__)

# Global state for tracking
_start_time: float = 0
_pipeline = None


def get_pipeline():
    """Get the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import TableExtractionPipeline
        _pipeline = TableExtractionPipeline(device=settings.model_device)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _start_time
    _start_time = time.time()
    
    logger.info("Starting Table Extraction API...")
    
    # Initialize pipeline (lazy load models) and preload model weights
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized")
        # Preload HuggingFace models to warm cache
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            logger.info("Preloading Table Transformer detection model...")
            _ = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
            _ = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
            logger.info("Preloading Table Transformer structure model...")
            _ = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
            _ = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        except Exception as me:
            logger.warning(f"Model preload skipped: {me}")
        # Initialize OCR reader to download models
        try:
            from src.ocr.easyocr_engine import EasyOCREngine
            ocr = EasyOCREngine(languages=settings.ocr_languages, device=pipeline.device)
            ocr.initialize()
            logger.info("OCR reader pre-initialized")
        except Exception as oe:
            logger.warning(f"OCR preload skipped: {oe}")
    except Exception as e:
        logger.warning(f"Pipeline initialization deferred: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Table Extraction API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Table Extraction API",
        description="""
        AI-powered OCR system for automatic table extraction from documents.
        
        ## Features
        - Extract tables from images and PDFs
        - Multi-language support (Korean, English)
        - Structure recognition with merged cell detection
        - Export to CSV, JSON, Excel, HTML
        
        ## Usage
        1. Upload a document using `/api/v1/extract`
        2. Get results in your preferred format
        3. Provide feedback to improve accuracy
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None,
                "status_code": 500
            }
        )
    
    # Include routers
    from .routes import router
    app.include_router(router, prefix="/api/v1")
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Root endpoint - serve the simple UI
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the simple table extraction UI."""
        static_file = Path(__file__).parent / "static" / "index.html"
        if static_file.exists():
            return HTMLResponse(content=static_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="""
        <html>
            <head><title>Table Extraction</title></head>
            <body>
                <h1>Table Extraction API</h1>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """)
    
    return app


def get_uptime() -> float:
    """Get application uptime in seconds."""
    global _start_time
    return time.time() - _start_time if _start_time > 0 else 0


# Create default app instance
app = create_app()
