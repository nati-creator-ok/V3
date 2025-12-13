"""
Image transformation functions for preprocessing.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def correct_skew(
    image: NDArray[np.uint8],
    max_angle: float = 45.0,
    angle_precision: float = 0.1
) -> Tuple[NDArray[np.uint8], float]:
    """
    Correct skew in document images using Hough transform.
    
    Args:
        image: Input image (grayscale or BGR)
        max_angle: Maximum angle to consider for deskewing
        angle_precision: Precision of angle detection in degrees
        
    Returns:
        Tuple of (deskewed image, detected angle in degrees)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180 * angle_precision,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return image, 0.0
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Normalize angle to [-45, 45] range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
            
        if abs(angle) <= max_angle:
            angles.append(angle)
    
    if not angles:
        return image, 0.0
    
    # Use median angle for robustness
    median_angle = np.median(angles)
    
    # Rotate image to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Calculate new bounding box size
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return deskewed, median_angle


def denoise(
    image: NDArray[np.uint8],
    method: str = "bilateral",
    strength: int = 10
) -> NDArray[np.uint8]:
    """
    Remove noise from document images.
    
    Args:
        image: Input image
        method: Denoising method ('bilateral', 'gaussian', 'median', 'nlmeans')
        strength: Denoising strength (interpretation depends on method)
        
    Returns:
        Denoised image
    """
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=strength * 7.5, sigmaSpace=strength * 7.5)
    
    elif method == "gaussian":
        kernel_size = max(3, (strength // 2) * 2 + 1)  # Ensure odd number
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == "median":
        kernel_size = max(3, (strength // 2) * 2 + 1)
        return cv2.medianBlur(image, kernel_size)
    
    elif method == "nlmeans":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def adaptive_binarize(
    image: NDArray[np.uint8],
    method: str = "otsu",
    block_size: int = 11,
    constant: int = 2
) -> NDArray[np.uint8]:
    """
    Convert image to binary using adaptive thresholding.
    
    Args:
        image: Input image (will be converted to grayscale)
        method: Binarization method ('otsu', 'adaptive_mean', 'adaptive_gaussian', 'sauvola')
        block_size: Block size for adaptive methods (must be odd)
        constant: Constant subtracted from mean/weighted mean
        
    Returns:
        Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    elif method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant
        )
    
    elif method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
        )
    
    elif method == "sauvola":
        # Sauvola binarization
        return _sauvola_threshold(gray, window_size=block_size, k=0.5)
    
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def _sauvola_threshold(
    image: NDArray[np.uint8],
    window_size: int = 15,
    k: float = 0.5,
    r: float = 128
) -> NDArray[np.uint8]:
    """Sauvola thresholding for document binarization."""
    # Compute local mean
    mean = cv2.blur(image.astype(np.float64), (window_size, window_size))
    
    # Compute local standard deviation
    sq_mean = cv2.blur(image.astype(np.float64) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    
    # Sauvola threshold
    threshold = mean * (1 + k * (std / r - 1))
    
    binary = np.zeros_like(image)
    binary[image > threshold] = 255
    
    return binary.astype(np.uint8)


def enhance_contrast(
    image: NDArray[np.uint8],
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> NDArray[np.uint8]:
    """
    Enhance image contrast.
    
    Args:
        image: Input image
        method: Enhancement method ('clahe', 'histogram', 'stretch')
        clip_limit: Clip limit for CLAHE
        tile_size: Tile grid size for CLAHE
        
    Returns:
        Contrast-enhanced image
    """
    if method == "clahe":
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            return clahe.apply(image)
    
    elif method == "histogram":
        if len(image.shape) == 3:
            # Equalize each channel
            channels = cv2.split(image)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            return cv2.merge(eq_channels)
        else:
            return cv2.equalizeHist(image)
    
    elif method == "stretch":
        # Linear contrast stretching
        min_val, max_val = np.percentile(image, (2, 98))
        stretched = np.clip((image.astype(np.float32) - min_val) * 255 / (max_val - min_val), 0, 255)
        return stretched.astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown contrast enhancement method: {method}")


def normalize_resolution(
    image: NDArray[np.uint8],
    target_dpi: int = 300,
    current_dpi: Optional[int] = None,
    max_size: int = 4096
) -> NDArray[np.uint8]:
    """
    Normalize image resolution for consistent OCR performance.
    
    Args:
        image: Input image
        target_dpi: Target DPI for normalization
        current_dpi: Current DPI (if known), otherwise estimated
        max_size: Maximum dimension size
        
    Returns:
        Resolution-normalized image
    """
    h, w = image.shape[:2]
    
    if current_dpi is None:
        # Estimate DPI based on image size (assume A4 document)
        # A4 at 300 DPI ≈ 2480 x 3508 pixels
        estimated_scale = max(w, h) / 3508
        current_dpi = int(300 * estimated_scale) or 150
    
    # Calculate scale factor
    scale = target_dpi / current_dpi
    
    # Apply scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Limit to max size
    if max(new_w, new_h) > max_size:
        ratio = max_size / max(new_w, new_h)
        new_w = int(new_w * ratio)
        new_h = int(new_h * ratio)
    
    # Resize using appropriate interpolation
    if scale > 1:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def remove_background(
    image: NDArray[np.uint8],
    method: str = "morphological"
) -> NDArray[np.uint8]:
    """
    Remove background noise and artifacts from document images.
    
    Args:
        image: Input image
        method: Background removal method ('morphological', 'flood_fill', 'grabcut')
        
    Returns:
        Image with background removed/cleaned
    """
    if method == "morphological":
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Morphological operations to estimate background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        background = cv2.GaussianBlur(background, (51, 51), 0)
        
        # Subtract background
        diff = cv2.absdiff(gray, background)
        
        # Normalize
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Invert if needed (text should be dark on light background)
        if np.mean(gray) < 128:
            normalized = 255 - normalized
        
        if len(image.shape) == 3:
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        return normalized
    
    elif method == "flood_fill":
        # Flood fill from corners to remove background
        result = image.copy()
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Fill from corners
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for corner in corners:
            cv2.floodFill(result, mask, corner, (255, 255, 255), (10, 10, 10), (10, 10, 10))
        
        return result
    
    else:
        raise ValueError(f"Unknown background removal method: {method}")


def auto_orient(image: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], int]:
    """
    Automatically detect and correct image orientation.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (oriented image, rotation applied in degrees)
    """
    # This is a placeholder - in production, you would use
    # a text orientation detection model or heuristics
    # For now, return the original image
    return image, 0
