#!/usr/bin/env python
"""
Script to download pre-trained model weights.
"""

import os
import argparse
import hashlib
from pathlib import Path
import requests
from tqdm import tqdm


# Model URLs and checksums
MODELS = {
    "yolov8_table": {
        "url": "https://example.com/models/yolov8_table_detection.pt",
        "filename": "yolov8_table_detection.pt",
        "checksum": "sha256:placeholder",
        "description": "YOLOv8 table detection model",
    },
    "table_transformer": {
        "url": "https://example.com/models/table_transformer.pt",
        "filename": "table_transformer.pt",
        "checksum": "sha256:placeholder",
        "description": "Table structure recognition transformer",
    },
    "detectron2_table": {
        "url": "https://example.com/models/detectron2_table.pkl",
        "filename": "detectron2_table.pkl",
        "checksum": "sha256:placeholder",
        "description": "Detectron2 table detection model",
    },
}

# Known public models
PUBLIC_MODELS = {
    "pubtables1m_detection": {
        "url": "https://huggingface.co/microsoft/table-transformer-detection/resolve/main/model.pth",
        "filename": "pubtables1m_detection.pth",
        "description": "Microsoft Table Transformer (PubTables-1M) - Detection",
    },
    "pubtables1m_structure": {
        "url": "https://huggingface.co/microsoft/table-transformer-structure-recognition/resolve/main/model.pth",
        "filename": "pubtables1m_structure.pth",
        "description": "Microsoft Table Transformer (PubTables-1M) - Structure Recognition",
    },
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify file checksum.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected checksum in format "algorithm:hash"
        
    Returns:
        True if checksum matches, False otherwise
    """
    if expected_checksum == "sha256:placeholder":
        print(f"Warning: No checksum verification for {file_path.name}")
        return True
    
    algorithm, expected_hash = expected_checksum.split(":")
    
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    actual_hash = hasher.hexdigest()
    return actual_hash == expected_hash


def download_model(
    model_name: str,
    output_dir: Path,
    force: bool = False,
    public: bool = False,
) -> bool:
    """
    Download a specific model.
    
    Args:
        model_name: Name of model to download
        output_dir: Output directory
        force: Force re-download even if exists
        public: Use public model sources
        
    Returns:
        True if successful, False otherwise
    """
    models = PUBLIC_MODELS if public else MODELS
    
    if model_name not in models:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(models.keys())}")
        return False
    
    model_info = models[model_name]
    output_path = output_dir / model_info["filename"]
    
    if output_path.exists() and not force:
        print(f"Model already exists: {output_path}")
        return True
    
    print(f"Downloading {model_info['description']}...")
    
    success = download_file(model_info["url"], output_path)
    
    if success and "checksum" in model_info:
        if not verify_checksum(output_path, model_info["checksum"]):
            print(f"Checksum verification failed for {output_path}")
            output_path.unlink()
            return False
    
    if success:
        print(f"Successfully downloaded to {output_path}")
    
    return success


def download_all(output_dir: Path, force: bool = False, public: bool = False):
    """
    Download all models.
    
    Args:
        output_dir: Output directory
        force: Force re-download
        public: Use public model sources
    """
    models = PUBLIC_MODELS if public else MODELS
    
    results = {}
    for model_name in models:
        results[model_name] = download_model(
            model_name, output_dir, force, public
        )
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model weights"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download (default: all)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Output directory for weights (default: models/weights)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    
    parser.add_argument(
        "--public",
        action="store_true",
        help="Download from public model hubs (HuggingFace, etc.)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nPrivate Models:")
        for name, info in MODELS.items():
            print(f"  - {name}: {info['description']}")
        
        print("\nPublic Models (--public):")
        for name, info in PUBLIC_MODELS.items():
            print(f"  - {name}: {info['description']}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model:
        download_model(args.model, output_dir, args.force, args.public)
    else:
        download_all(output_dir, args.force, args.public)


if __name__ == "__main__":
    main()
