"""
Data models for table structure recognition.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CellType(Enum):
    """Type of table cell."""
    DATA = "data"
    HEADER = "header"
    INDEX = "index"
    EMPTY = "empty"


class CellRelation(Enum):
    """Spatial relation between cells."""
    SAME_ROW = "same_row"
    SAME_COLUMN = "same_column"
    NO_RELATION = "no_relation"


@dataclass
class Cell:
    """Represents a single cell in a table."""
    
    row: int  # Row index (0-based)
    col: int  # Column index (0-based)
    rowspan: int = 1  # Number of rows spanned
    colspan: int = 1  # Number of columns spanned
    text: str = ""  # Cell text content
    confidence: float = 1.0  # Recognition confidence
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    cell_type: CellType = CellType.DATA
    
    @property
    def end_row(self) -> int:
        """Last row index (exclusive)."""
        return self.row + self.rowspan
    
    @property
    def end_col(self) -> int:
        """Last column index (exclusive)."""
        return self.col + self.colspan
    
    @property
    def is_merged(self) -> bool:
        """Check if cell spans multiple rows/columns."""
        return self.rowspan > 1 or self.colspan > 1
    
    @property
    def is_header(self) -> bool:
        """Check if cell is a header."""
        return self.cell_type == CellType.HEADER
    
    def overlaps(self, other: "Cell") -> bool:
        """Check if this cell overlaps with another cell."""
        return not (
            self.end_row <= other.row or
            other.end_row <= self.row or
            self.end_col <= other.col or
            other.end_col <= self.col
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox) if self.bbox else None,
            "cell_type": self.cell_type.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cell":
        """Create from dictionary."""
        return cls(
            row=data["row"],
            col=data["col"],
            rowspan=data.get("rowspan", 1),
            colspan=data.get("colspan", 1),
            text=data.get("text", ""),
            confidence=data.get("confidence", 1.0),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            cell_type=CellType(data.get("cell_type", "data"))
        )


@dataclass
class TableStructure:
    """Represents the complete structure of a table."""
    
    cells: List[Cell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None  # Table bounding box
    confidence: float = 1.0
    has_header: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate dimensions if not provided."""
        if self.cells and (self.num_rows == 0 or self.num_cols == 0):
            self._calculate_dimensions()
    
    def _calculate_dimensions(self) -> None:
        """Calculate number of rows and columns from cells."""
        if not self.cells:
            return
        
        max_row = max(cell.end_row for cell in self.cells)
        max_col = max(cell.end_col for cell in self.cells)
        
        self.num_rows = max_row
        self.num_cols = max_col
    
    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get cell at specified position (considering merged cells)."""
        for cell in self.cells:
            if (cell.row <= row < cell.end_row and
                cell.col <= col < cell.end_col):
                return cell
        return None
    
    def get_row(self, row: int) -> List[Cell]:
        """Get all cells in a row."""
        return [cell for cell in self.cells if cell.row == row]
    
    def get_column(self, col: int) -> List[Cell]:
        """Get all cells in a column."""
        return [cell for cell in self.cells if cell.col == col]
    
    def get_header_row(self) -> List[Cell]:
        """Get header row cells (first row or cells marked as header)."""
        header_cells = [c for c in self.cells if c.cell_type == CellType.HEADER]
        if header_cells:
            return header_cells
        return self.get_row(0)
    
    def to_2d_array(self) -> List[List[str]]:
        """Convert to 2D array of text values."""
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        
        for cell in self.cells:
            for r in range(cell.row, cell.end_row):
                for c in range(cell.col, cell.end_col):
                    if r < self.num_rows and c < self.num_cols:
                        grid[r][c] = cell.text
        
        return grid
    
    def to_html(self, include_style: bool = True) -> str:
        """Convert table structure to HTML."""
        # Track which cells have been rendered
        rendered = set()
        
        html_parts = []
        
        if include_style:
            html_parts.append(
                '<table border="1" style="border-collapse: collapse;">'
            )
        else:
            html_parts.append("<table>")
        
        for row in range(self.num_rows):
            html_parts.append("  <tr>")
            
            for col in range(self.num_cols):
                # Skip if this position was covered by a merged cell
                if (row, col) in rendered:
                    continue
                
                cell = self.get_cell(row, col)
                if cell is None:
                    html_parts.append("    <td></td>")
                    continue
                
                # Mark all positions covered by this cell
                for r in range(cell.row, cell.end_row):
                    for c in range(cell.col, cell.end_col):
                        rendered.add((r, c))
                
                # Build cell tag
                tag = "th" if cell.is_header else "td"
                attrs = []
                
                if cell.rowspan > 1:
                    attrs.append(f'rowspan="{cell.rowspan}"')
                if cell.colspan > 1:
                    attrs.append(f'colspan="{cell.colspan}"')
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                html_parts.append(f"    <{tag}{attr_str}>{cell.text}</{tag}>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def to_csv(self, delimiter: str = ",") -> str:
        """Convert table structure to CSV string."""
        import csv
        import io
        
        grid = self.to_2d_array()
        
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        writer.writerows(grid)
        
        return output.getvalue()
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        
        grid = self.to_2d_array()
        
        if self.has_header and len(grid) > 0:
            return pd.DataFrame(grid[1:], columns=grid[0])
        else:
            return pd.DataFrame(grid)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "cells": [cell.to_dict() for cell in self.cells],
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence": self.confidence,
            "has_header": self.has_header,
            "html": self.to_html(include_style=False),
            "csv": self.to_csv()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableStructure":
        """Create from dictionary."""
        cells = [Cell.from_dict(c) for c in data.get("cells", [])]
        return cls(
            cells=cells,
            num_rows=data.get("num_rows", 0),
            num_cols=data.get("num_cols", 0),
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            confidence=data.get("confidence", 1.0),
            has_header=data.get("has_header", False)
        )
    
    @classmethod
    def from_2d_array(
        cls,
        data: List[List[str]],
        has_header: bool = False
    ) -> "TableStructure":
        """Create from 2D array of text values."""
        cells = []
        num_rows = len(data)
        num_cols = max(len(row) for row in data) if data else 0
        
        for row_idx, row in enumerate(data):
            for col_idx, text in enumerate(row):
                cell_type = CellType.HEADER if (has_header and row_idx == 0) else CellType.DATA
                cells.append(Cell(
                    row=row_idx,
                    col=col_idx,
                    text=str(text),
                    cell_type=cell_type
                ))
        
        return cls(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
            has_header=has_header
        )


@dataclass 
class StructureRecognitionResult:
    """Result from structure recognition."""
    
    table: TableStructure
    inference_time: float = 0.0
    model_name: str = ""
    raw_predictions: Optional[Dict[str, Any]] = None
