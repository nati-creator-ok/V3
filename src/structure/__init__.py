"""
Table structure recognition module.
Detects cells, rows, columns, and merged cell regions.
"""

from .recognizer import TableStructureRecognizer
from .models import Cell, TableStructure, CellRelation
from .transformer_recognizer import TransformerTableRecognizer

__all__ = [
    "TableStructureRecognizer",
    "TransformerTableRecognizer",
    "Cell",
    "TableStructure",
    "CellRelation",
]
