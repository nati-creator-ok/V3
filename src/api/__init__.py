"""
FastAPI REST API for Table Extraction System.
"""

from .app import create_app, app
from .routes import router

__all__ = ["create_app", "app", "router"]
