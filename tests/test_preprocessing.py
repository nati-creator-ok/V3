"""
Tests for preprocessing module.
"""

import numpy as np
import pytest

from src.preprocessing import (
    PreprocessingPipeline,
    PreprocessingConfig,
    correct_skew,
    denoise,
    adaptive_binarize,
    enhance_contrast,
    normalize_resolution,
)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple grayscale image with some noise
    img = np.ones((500, 400), dtype=np.uint8) * 255
    # Add some black rectangles (simulating text/table)
    img[100:150, 50:350] = 0
    img[200:250, 50:350] = 0
    img[300:350, 50:350] = 0
    # Add noise
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) - noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def color_image():
    """Create a sample color image."""
    img = np.ones((500, 400, 3), dtype=np.uint8) * 255
    img[100:150, 50:350] = [0, 0, 0]
    img[200:250, 50:350] = [0, 0, 0]
    return img


class TestDenoise:
    """Test denoising functions."""
    
    def test_bilateral_denoise(self, sample_image):
        result = denoise(sample_image, method="bilateral")
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_gaussian_denoise(self, sample_image):
        result = denoise(sample_image, method="gaussian")
        assert result.shape == sample_image.shape
    
    def test_median_denoise(self, sample_image):
        result = denoise(sample_image, method="median")
        assert result.shape == sample_image.shape
    
    def test_nlmeans_denoise(self, sample_image):
        result = denoise(sample_image, method="nlmeans")
        assert result.shape == sample_image.shape
    
    def test_invalid_method(self, sample_image):
        with pytest.raises(ValueError):
            denoise(sample_image, method="invalid")


class TestBinarize:
    """Test binarization functions."""
    
    def test_otsu_binarize(self, sample_image):
        result = adaptive_binarize(sample_image, method="otsu")
        assert result.shape == sample_image.shape
        assert set(np.unique(result)).issubset({0, 255})
    
    def test_adaptive_mean_binarize(self, sample_image):
        result = adaptive_binarize(sample_image, method="adaptive_mean")
        assert result.shape == sample_image.shape
    
    def test_adaptive_gaussian_binarize(self, sample_image):
        result = adaptive_binarize(sample_image, method="adaptive_gaussian")
        assert result.shape == sample_image.shape
    
    def test_sauvola_binarize(self, sample_image):
        result = adaptive_binarize(sample_image, method="sauvola")
        assert result.shape == sample_image.shape
    
    def test_color_image_binarize(self, color_image):
        result = adaptive_binarize(color_image, method="otsu")
        assert len(result.shape) == 2  # Should be grayscale


class TestContrast:
    """Test contrast enhancement functions."""
    
    def test_clahe_grayscale(self, sample_image):
        result = enhance_contrast(sample_image, method="clahe")
        assert result.shape == sample_image.shape
    
    def test_clahe_color(self, color_image):
        result = enhance_contrast(color_image, method="clahe")
        assert result.shape == color_image.shape
    
    def test_histogram_equalization(self, sample_image):
        result = enhance_contrast(sample_image, method="histogram")
        assert result.shape == sample_image.shape
    
    def test_stretch(self, sample_image):
        result = enhance_contrast(sample_image, method="stretch")
        assert result.shape == sample_image.shape


class TestResolution:
    """Test resolution normalization."""
    
    def test_upscale(self, sample_image):
        result = normalize_resolution(sample_image, target_dpi=600, current_dpi=300)
        assert result.shape[0] > sample_image.shape[0]
        assert result.shape[1] > sample_image.shape[1]
    
    def test_downscale(self, sample_image):
        result = normalize_resolution(sample_image, target_dpi=150, current_dpi=300)
        assert result.shape[0] < sample_image.shape[0]
        assert result.shape[1] < sample_image.shape[1]
    
    def test_max_size_limit(self, sample_image):
        result = normalize_resolution(
            sample_image, target_dpi=1200, current_dpi=300, max_size=1000
        )
        assert max(result.shape) <= 1000


class TestPipeline:
    """Test preprocessing pipeline."""
    
    def test_default_pipeline(self, color_image):
        pipeline = PreprocessingPipeline()
        result = pipeline.process(color_image)
        
        assert result.image is not None
        assert len(result.applied_transforms) > 0
    
    def test_custom_config(self, sample_image):
        config = PreprocessingConfig(
            enable_deskew=False,
            enable_denoise=True,
            enable_binarize=True,
            denoise_method="median"
        )
        pipeline = PreprocessingPipeline(config=config)
        result = pipeline.process(sample_image)
        
        assert "binarization" in result.applied_transforms
        assert "denoising" in result.applied_transforms
        assert "deskew" not in result.applied_transforms
    
    def test_from_dict(self, sample_image):
        config_dict = {
            "enable_deskew": True,
            "enable_denoise": False,
            "enable_binarize": False
        }
        pipeline = PreprocessingPipeline.from_dict(config_dict)
        result = pipeline.process(sample_image)
        
        assert "denoising" not in result.applied_transforms
    
    def test_to_dict(self):
        pipeline = PreprocessingPipeline()
        config_dict = pipeline.to_dict()
        
        assert "enable_deskew" in config_dict
        assert "enable_denoise" in config_dict
        assert "target_dpi" in config_dict
