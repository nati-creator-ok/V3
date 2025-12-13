"""
Tesseract OCR engine implementation.
"""

import logging
import time
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .engine import OCREngine
from .models import OCRResult, TextBox

logger = logging.getLogger(__name__)


class TesseractEngine(OCREngine):
    """
    OCR engine using Tesseract OCR.
    
    Tesseract is a widely-used open-source OCR engine.
    Requires tesseract to be installed on the system.
    
    Example:
        engine = TesseractEngine(languages=['kor', 'eng'])
        result = engine.recognize(image)
    """
    
    # Language code mapping to Tesseract codes
    LANGUAGE_MAP = {
        "ko": "kor",
        "korean": "kor",
        "en": "eng",
        "english": "eng",
        "ja": "jpn",
        "japanese": "jpn",
        "zh": "chi_sim",
        "chinese": "chi_sim",
    }
    
    def __init__(
        self,
        languages: Optional[List[str]] = None,
        device: str = "cpu",  # Tesseract runs on CPU
        confidence_threshold: float = 0.0,
        psm: int = 6,  # Page segmentation mode
        oem: int = 3   # OCR Engine mode
    ):
        """
        Initialize Tesseract engine.
        
        Args:
            languages: List of language codes
            device: Device (ignored, Tesseract uses CPU)
            confidence_threshold: Minimum confidence threshold
            psm: Page segmentation mode (0-13)
            oem: OCR Engine mode (0-3)
        """
        super().__init__(
            languages=languages or ["eng"],
            device="cpu",
            confidence_threshold=confidence_threshold
        )
        self.psm = psm
        self.oem = oem
    
    def initialize(self) -> None:
        """Verify Tesseract installation."""
        try:
            import pytesseract
            # Test that tesseract is accessible
            pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError(
                "pytesseract package is required. Install with: pip install pytesseract"
            )
        except Exception as e:
            raise RuntimeError(
                f"Tesseract OCR not found. Please install Tesseract: {e}"
            )
        
        self._is_initialized = True
        logger.info("Tesseract OCR initialized")
    
    def _recognize(
        self,
        image: NDArray[np.uint8],
        detail: bool = True
    ) -> OCRResult:
        """Run Tesseract recognition."""
        import pytesseract
        
        start_time = time.time()
        h, w = image.shape[:2]
        
        # Build language string
        lang_codes = []
        for lang in self.languages:
            mapped = self.LANGUAGE_MAP.get(lang.lower(), lang)
            if mapped not in lang_codes:
                lang_codes.append(mapped)
        lang_string = "+".join(lang_codes)
        
        # Configure Tesseract
        config = f"--psm {self.psm} --oem {self.oem}"
        
        if detail:
            # Get detailed output with bounding boxes
            data = pytesseract.image_to_data(
                image,
                lang=lang_string,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            text_boxes = []
            full_texts = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Skip empty text or low confidence
                if not text or conf < 0:
                    continue
                
                x = data['left'][i]
                y = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                confidence = conf / 100.0
                
                text_boxes.append(TextBox(
                    text=text,
                    confidence=confidence,
                    bbox=(float(x), float(y), float(x + width), float(y + height))
                ))
                full_texts.append(text)
            
            full_text = " ".join(full_texts)
        else:
            # Simple text extraction
            full_text = pytesseract.image_to_string(
                image,
                lang=lang_string,
                config=config
            ).strip()
            text_boxes = []
        
        inference_time = time.time() - start_time
        
        return OCRResult(
            text_boxes=text_boxes,
            full_text=full_text,
            inference_time=inference_time,
            engine_name="Tesseract",
            languages=self.languages,
            image_width=w,
            image_height=h,
            metadata={"psm": self.psm, "oem": self.oem}
        )
    
    def get_hocr(self, image: NDArray[np.uint8]) -> str:
        """
        Get hOCR output (HTML-like format with layout info).
        
        Args:
            image: Input image
            
        Returns:
            hOCR string
        """
        import pytesseract
        
        if not self._is_initialized:
            self.initialize()
        
        lang_codes = [self.LANGUAGE_MAP.get(l.lower(), l) for l in self.languages]
        lang_string = "+".join(lang_codes)
        
        return pytesseract.image_to_pdf_or_hocr(
            image,
            lang=lang_string,
            extension='hocr',
            config=f"--psm {self.psm} --oem {self.oem}"
        ).decode('utf-8')
    
    def recognize_table(self, image: NDArray[np.uint8]) -> str:
        """
        Recognize text from table image using table-specific PSM.
        
        Args:
            image: Table image
            
        Returns:
            Recognized text preserving some table structure
        """
        import pytesseract
        
        if not self._is_initialized:
            self.initialize()
        
        lang_codes = [self.LANGUAGE_MAP.get(l.lower(), l) for l in self.languages]
        lang_string = "+".join(lang_codes)
        
        # PSM 6: Assume a single uniform block of text
        # PSM 4: Assume a single column of text of variable sizes
        config = f"--psm 6 --oem {self.oem}"
        
        return pytesseract.image_to_string(
            image,
            lang=lang_string,
            config=config
        ).strip()
