"""
Preprocessing pipeline that chains multiple transforms.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from pdf2image import convert_from_path

from .transforms import (
    adaptive_binarize,
    correct_skew,
    denoise,
    enhance_contrast,
    normalize_resolution,
    remove_background,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Deskew settings
    enable_deskew: bool = True
    max_skew_angle: float = 45.0
    
    # Denoising settings
    enable_denoise: bool = True
    denoise_method: str = "bilateral"
    denoise_strength: int = 10
    
    # Binarization settings
    enable_binarize: bool = False  # Off by default, not always needed
    binarize_method: str = "adaptive_gaussian"
    binarize_block_size: int = 11
    
    # Contrast enhancement
    enable_contrast: bool = True
    contrast_method: str = "clahe"
    clahe_clip_limit: float = 2.0
    
    # Resolution normalization
    enable_resolution_norm: bool = True
    target_dpi: int = 300
    max_image_size: int = 4096
    
    # Background removal
    enable_background_removal: bool = False
    background_method: str = "morphological"
    
    # PDF conversion settings
    pdf_dpi: int = 300
    pdf_fmt: str = "RGB"


@dataclass
class PreprocessingResult:
    """Result of preprocessing a single image."""
    
    image: NDArray[np.uint8]
    original_size: tuple
    final_size: tuple
    skew_angle: float = 0.0
    applied_transforms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for document images.
    
    Example:
        pipeline = PreprocessingPipeline()
        result = pipeline.process("document.pdf")
        cv2.imwrite("preprocessed.png", result.image)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or PreprocessingConfig()
    
    def process(
        self,
        input_data: Union[str, Path, NDArray[np.uint8]],
        page_number: int = 0
    ) -> PreprocessingResult:
        """
        Process a single image or PDF page.
        
        Args:
            input_data: File path (str/Path) or numpy array
            page_number: Page number for PDFs (0-indexed)
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        # Load image
        if isinstance(input_data, (str, Path)):
            image = self._load_image(input_data, page_number)
        else:
            image = input_data.copy()
        
        original_size = image.shape[:2]
        applied_transforms = []
        skew_angle = 0.0
        
        # Apply transforms in order
        
        # 1. Resolution normalization (first, to standardize input)
        if self.config.enable_resolution_norm:
            image = normalize_resolution(
                image,
                target_dpi=self.config.target_dpi,
                max_size=self.config.max_image_size
            )
            applied_transforms.append("resolution_normalization")
        
        # 2. Contrast enhancement
        if self.config.enable_contrast:
            image = enhance_contrast(
                image,
                method=self.config.contrast_method,
                clip_limit=self.config.clahe_clip_limit
            )
            applied_transforms.append("contrast_enhancement")
        
        # 3. Denoising
        if self.config.enable_denoise:
            image = denoise(
                image,
                method=self.config.denoise_method,
                strength=self.config.denoise_strength
            )
            applied_transforms.append("denoising")
        
        # 4. Deskewing
        if self.config.enable_deskew:
            image, skew_angle = correct_skew(
                image,
                max_angle=self.config.max_skew_angle
            )
            applied_transforms.append("deskew")
        
        # 5. Background removal
        if self.config.enable_background_removal:
            image = remove_background(
                image,
                method=self.config.background_method
            )
            applied_transforms.append("background_removal")
        
        # 6. Binarization (optional, last step)
        if self.config.enable_binarize:
            image = adaptive_binarize(
                image,
                method=self.config.binarize_method,
                block_size=self.config.binarize_block_size
            )
            applied_transforms.append("binarization")
        
        return PreprocessingResult(
            image=image,
            original_size=original_size,
            final_size=image.shape[:2],
            skew_angle=skew_angle,
            applied_transforms=applied_transforms
        )
    
    def process_batch(
        self,
        inputs: List[Union[str, Path, NDArray[np.uint8]]]
    ) -> List[PreprocessingResult]:
        """
        Process multiple images.
        
        Args:
            inputs: List of file paths or numpy arrays
            
        Returns:
            List of PreprocessingResult objects
        """
        results = []
        for idx, input_data in enumerate(inputs):
            try:
                result = self.process(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing input {idx}: {e}")
                raise
        return results
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> List[PreprocessingResult]:
        """
        Process all pages of a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PreprocessingResult objects, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=self.config.pdf_dpi,
            fmt=self.config.pdf_fmt
        )
        
        results = []
        for idx, pil_image in enumerate(images):
            # Convert PIL to numpy array
            image = np.array(pil_image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            result = self.process(image)
            result.metadata["page_number"] = idx
            result.metadata["source_file"] = str(pdf_path)
            results.append(result)
        
        return results
    
    def _load_image(
        self,
        path: Union[str, Path],
        page_number: int = 0
    ) -> NDArray[np.uint8]:
        """Load image from file path."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            # Load specific page from PDF
            images = convert_from_path(
                path,
                dpi=self.config.pdf_dpi,
                fmt=self.config.pdf_fmt,
                first_page=page_number + 1,
                last_page=page_number + 1
            )
            if not images:
                raise ValueError(f"Could not load page {page_number} from PDF")
            
            image = np.array(images[0])
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        
        else:
            # Load regular image
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            return image
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PreprocessingPipeline":
        """Create pipeline from configuration dictionary."""
        config = PreprocessingConfig(**config_dict)
        return cls(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "enable_deskew": self.config.enable_deskew,
            "max_skew_angle": self.config.max_skew_angle,
            "enable_denoise": self.config.enable_denoise,
            "denoise_method": self.config.denoise_method,
            "denoise_strength": self.config.denoise_strength,
            "enable_binarize": self.config.enable_binarize,
            "binarize_method": self.config.binarize_method,
            "enable_contrast": self.config.enable_contrast,
            "contrast_method": self.config.contrast_method,
            "enable_resolution_norm": self.config.enable_resolution_norm,
            "target_dpi": self.config.target_dpi,
            "max_image_size": self.config.max_image_size,
        }
