"""
Download and cache Microsoft Table Transformer models for offline use.
This pre-downloads the models so the server starts faster.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_models():
    """Download all required models from HuggingFace."""
    print("="*60)
    print("Downloading AI Models for Table Extraction")
    print("="*60)
    print("\nThis will download ~600MB of model weights.")
    print("Models will be cached in: C:\\Users\\YOGA\\.cache\\huggingface\\")
    print("\n")
    
    try:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        import torch
        
        # 1. Download Table Detection Model
        print("📥 [1/3] Downloading Table Detection Model...")
        print("    Model: microsoft/table-transformer-detection")
        detector_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        detector_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        print("    ✅ Table Detection Model downloaded!\n")
        
        # 2. Download Table Structure Recognition Model
        print("📥 [2/3] Downloading Table Structure Model...")
        print("    Model: microsoft/table-transformer-structure-recognition")
        structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        print("    ✅ Table Structure Model downloaded!\n")
        
        # 3. Download EasyOCR Models
        print("📥 [3/3] Downloading EasyOCR Models (Korean + English)...")
        print("    Languages: Korean, English")
        import easyocr
        reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        print("    ✅ EasyOCR Models downloaded!\n")
        
        print("="*60)
        print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print("\n📌 Models are now cached. Server will start much faster!")
        print("\nNext steps:")
        print("  1. Run: uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload")
        print("  2. Open: http://127.0.0.1:8000")
        print("\n")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Error: Missing required packages")
        print(f"   {e}")
        print("\n💡 Install dependencies first:")
        print("   pip install -r requirements.txt\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        print("\n💡 Check your internet connection and try again.\n")
        return False


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
