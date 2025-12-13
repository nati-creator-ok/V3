"""
Test configuration for pytest.
"""

import pytest
import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_table_cells():
    """Create sample table cells for testing."""
    return [
        {"row": 0, "col": 0, "text": "Header 1", "rowspan": 1, "colspan": 1},
        {"row": 0, "col": 1, "text": "Header 2", "rowspan": 1, "colspan": 1},
        {"row": 1, "col": 0, "text": "Value A", "rowspan": 1, "colspan": 1},
        {"row": 1, "col": 1, "text": "Value B", "rowspan": 1, "colspan": 1},
    ]


@pytest.fixture
def sample_table_html():
    """Create sample HTML table."""
    return """
    <table>
        <tr>
            <td>Header 1</td>
            <td>Header 2</td>
        </tr>
        <tr>
            <td>Value A</td>
            <td>Value B</td>
        </tr>
    </table>
    """


@pytest.fixture
def mock_detection_result():
    """Create mock detection result."""
    from src.detection.models import DetectionResult, BoundingBox
    
    return DetectionResult(
        boxes=[
            BoundingBox(
                x1=100, y1=100, x2=500, y2=400,
                confidence=0.95,
                class_name="table"
            )
        ],
        confidence_scores=[0.95],
        class_names=["table"],
        image_size=(800, 600)
    )


@pytest.fixture
def mock_structure_result():
    """Create mock structure recognition result."""
    from src.structure.models import (
        StructureResult, Cell, TableStructure
    )
    
    return StructureResult(
        cells=[
            Cell(row=0, col=0, row_span=1, col_span=1,
                 bbox=(100, 100, 200, 150), text="A"),
            Cell(row=0, col=1, row_span=1, col_span=1,
                 bbox=(200, 100, 300, 150), text="B"),
        ],
        structure=TableStructure(
            num_rows=1,
            num_cols=2,
            has_header=True
        ),
        html="<table><tr><td>A</td><td>B</td></tr></table>",
        confidence=0.9
    )


@pytest.fixture
def mock_ocr_result():
    """Create mock OCR result."""
    from src.ocr.models import OCRResult, TextBlock
    
    return OCRResult(
        text="Sample text",
        blocks=[
            TextBlock(
                text="Sample",
                bbox=(0, 0, 50, 20),
                confidence=0.95,
                language="en"
            ),
            TextBlock(
                text="text",
                bbox=(55, 0, 100, 20),
                confidence=0.92,
                language="en"
            ),
        ],
        language="en",
        confidence=0.94
    )
