"""
Command-line interface for Table Extraction System.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from src import __version__
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="table-extract")
def cli():
    """AI OCR-based Automatic Table Extraction System."""
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for extracted tables"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["csv", "json", "excel", "html", "all"]),
    default="csv",
    help="Output format"
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default="cuda",
    help="Device for inference"
)
@click.option(
    "--languages", "-l",
    default="ko,en",
    help="OCR languages (comma-separated)"
)
@click.option(
    "--confidence",
    type=float,
    default=0.5,
    help="Confidence threshold"
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
    help="Skip image preprocessing"
)
@click.option(
    "--skip-ocr",
    is_flag=True,
    help="Skip OCR (structure only)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
def extract(
    input_path: str,
    output: str,
    format: str,
    device: str,
    languages: str,
    confidence: float,
    skip_preprocessing: bool,
    skip_ocr: bool,
    verbose: bool
):
    """
    Extract tables from document image or PDF.
    
    Example:
        table-extract extract document.pdf -o results/ -f csv
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    from src.pipeline import TableExtractionPipeline
    
    input_path = Path(input_path)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Processing: {input_path}")
    click.echo(f"Device: {device}")
    click.echo(f"Languages: {languages}")
    
    # Initialize pipeline
    pipeline = TableExtractionPipeline(device=device)
    
    # Process input
    if input_path.suffix.lower() == ".pdf":
        results = pipeline.extract_from_pdf(input_path)
    else:
        results = [pipeline.extract(
            str(input_path),
            skip_preprocessing=skip_preprocessing,
            skip_ocr=skip_ocr
        )]
    
    # Export results
    for idx, result in enumerate(results):
        base_name = f"{input_path.stem}_page{idx}" if len(results) > 1 else input_path.stem
        
        click.echo(f"\nPage {idx + 1}: Found {result.num_tables} table(s)")
        click.echo(f"Processing time: {result.total_time:.2f}s")
        
        for t_idx, table in enumerate(result.tables):
            table_name = f"{base_name}_table{t_idx + 1}"
            
            if format in ("csv", "all"):
                csv_path = output_path / f"{table_name}.csv"
                csv_path.write_text(table.to_csv(), encoding="utf-8")
                click.echo(f"  Saved: {csv_path}")
            
            if format in ("html", "all"):
                html_path = output_path / f"{table_name}.html"
                html_path.write_text(table.to_html(), encoding="utf-8")
                click.echo(f"  Saved: {html_path}")
            
            if format in ("json", "all"):
                import json
                json_path = output_path / f"{table_name}.json"
                json_path.write_text(
                    json.dumps(table.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                click.echo(f"  Saved: {json_path}")
            
            if format in ("excel", "all"):
                excel_path = output_path / f"{table_name}.xlsx"
                df = table.to_dataframe()
                df.to_excel(excel_path, index=False)
                click.echo(f"  Saved: {excel_path}")
    
    click.echo(f"\n✓ Extraction complete!")


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind"
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind"
)
@click.option(
    "--workers",
    default=4,
    type=int,
    help="Number of workers"
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload (development)"
)
def serve(host: str, port: int, workers: int, reload: bool):
    """
    Start the API server.
    
    Example:
        table-extract serve --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    click.echo(f"Starting server at http://{host}:{port}")
    click.echo(f"API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.argument("annotations_path", type=click.Path(exists=True))
@click.argument("images_dir", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="evaluation_report.json",
    help="Output path for report"
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default="cuda",
    help="Device for inference"
)
def evaluate(
    annotations_path: str,
    images_dir: str,
    output: str,
    device: str
):
    """
    Run evaluation on test dataset.
    
    Example:
        table-extract evaluate annotations.json images/ -o report.json
    """
    from src.pipeline import TableExtractionPipeline
    from src.evaluation import TableExtractionEvaluator
    
    click.echo("Loading pipeline...")
    pipeline = TableExtractionPipeline(device=device)
    
    click.echo("Loading dataset...")
    evaluator = TableExtractionEvaluator(pipeline)
    dataset = evaluator.load_dataset(annotations_path, images_dir)
    
    click.echo(f"Evaluating {len(dataset)} samples...")
    report = evaluator.evaluate(dataset, verbose=True)
    
    click.echo("\n" + str(report))
    
    evaluator.save_report(report, output)
    click.echo(f"\nReport saved to: {output}")


@cli.command()
@click.option(
    "--model",
    type=click.Choice(["detection", "structure", "all"]),
    default="all",
    help="Model to train"
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Training data path"
)
@click.option(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs"
)
@click.option(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="./checkpoints",
    help="Output directory for checkpoints"
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default="cuda",
    help="Device for training"
)
def train(
    model: str,
    data: str,
    epochs: int,
    batch_size: int,
    output: str,
    device: str
):
    """
    Train models on custom dataset.
    
    Example:
        table-extract train --model detection --data data.yaml --epochs 100
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if model in ("detection", "all"):
        click.echo("Training detection model...")
        from src.detection import YOLOTableDetector
        
        detector = YOLOTableDetector(device=device)
        detector.load_model()
        
        results = detector.train(
            data_yaml=data,
            epochs=epochs,
            batch_size=batch_size,
            project=str(output_path / "detection")
        )
        click.echo("Detection model training complete!")
    
    if model in ("structure", "all"):
        click.echo("Training structure model...")
        # Structure model training would go here
        click.echo("Structure model training not yet implemented")
    
    click.echo(f"\nCheckpoints saved to: {output}")


@cli.command()
def info():
    """Show system and configuration information."""
    import torch
    
    click.echo(f"Table Extraction System v{__version__}")
    click.echo(f"\nConfiguration:")
    click.echo(f"  Environment: {settings.app_env}")
    click.echo(f"  Debug: {settings.debug}")
    click.echo(f"  Device: {settings.model_device}")
    click.echo(f"  OCR Languages: {settings.ocr_languages}")
    
    click.echo(f"\nSystem:")
    click.echo(f"  Python: {sys.version}")
    click.echo(f"  PyTorch: {torch.__version__}")
    click.echo(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        click.echo(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        click.echo(f"  CUDA Version: {torch.version.cuda}")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
