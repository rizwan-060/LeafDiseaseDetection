"""
Image Preprocessing Module for Leaf Disease Detection.
Handles resizing, normalization, histogram equalization, and leaf segmentation
to standardize input before model inference.
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import cv2

import config


def resize_image(image: np.ndarray, target_size: tuple = config.IMG_SIZE) -> np.ndarray:
    """
    Resizes an image to the target dimensions using bilinear interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C).
        target_size: Tuple (width, height) to resize to.
        
    Returns:
        np.ndarray: Resized image with shape (target_h, target_w, 3).
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_image(image: np.ndarray, method: str = "scale_255") -> np.ndarray:
    """
    Normalizes pixel values to a standard range.
    
    Args:
        image: Input image as numpy array.
        method: "scale_255" for [0,1], or "scale_127" for [-1,1].
        
    Returns:
        np.ndarray: Normalized float32 image.
    """
    image = image.astype(np.float32)
    
    if method == "scale_255":
        return image / 255.0
    elif method == "scale_127":
        return (image / 127.5) - 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to improve contrast.
    Operates on the Y channel of YUV color space to preserve color.
    
    Args:
        image: Input image in RGB format (H, W, 3).
        
    Returns:
        np.ndarray: Contrast-enhanced image in RGB.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Convert RGB to YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # Equalize the Y (luminance) channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # Convert back to RGB
    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return equalized


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Attempts to segment the leaf from the background using color thresholding.
    This is a lightweight heuristic; complex backgrounds may need GrabCut.
    
    Args:
        image: Input image in RGB format.
        
    Returns:
        np.ndarray: Segmented image with black background where leaf was not detected.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Green color range in HSV (broad to catch various leaf shades)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented


def preprocess_image(
    image_path: Union[str, Path],
    apply_equalization: bool = True,
    apply_segmentation: bool = False,
    normalize_method: str = "scale_255"
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image file.
    
    Pipeline:
        1. Load image (BGR -> RGB)
        2. Resize to IMG_SIZE
        3. Optional: Histogram equalization
        4. Optional: Leaf segmentation
        5. Normalize pixel values
        
    Args:
        image_path: Path to image file.
        apply_equalization: Whether to apply histogram equalization.
        apply_segmentation: Whether to apply leaf segmentation.
        normalize_method: "scale_255" or "scale_127".
        
    Returns:
        np.ndarray: Preprocessed image ready for model input.
        
    Raises:
        FileNotFoundError: If image cannot be loaded.
        ValueError: If image is corrupted or invalid.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = resize_image(image, config.IMG_SIZE)
    
    # Optional: Enhance contrast
    if apply_equalization:
        image = apply_histogram_equalization(image)
    
    # Optional: Remove background
    if apply_segmentation:
        image = segment_leaf(image)
    
    # Normalize
    image = normalize_image(image, method=normalize_method)
    
    return image


def preprocess_array(
    image: np.ndarray,
    apply_equalization: bool = True,
    apply_segmentation: bool = False,
    normalize_method: str = "scale_255"
) -> np.ndarray:
    """
    Full preprocessing pipeline for an in-memory numpy array.
    Same as preprocess_image but accepts an array instead of a path.
    
    Args:
        image: Input image in RGB format.
        
    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image array is empty or None")
    
    image = resize_image(image, config.IMG_SIZE)
    
    if apply_equalization:
        image = apply_histogram_equalization(image)
    
    if apply_segmentation:
        image = segment_leaf(image)
    
    image = normalize_image(image, method=normalize_method)
    return image


def add_batch_dimension(image: np.ndarray) -> np.ndarray:
    """
    Adds a batch dimension to a single image for model prediction.
    (224, 224, 3) -> (1, 224, 224, 3)
    
    Args:
        image: Preprocessed image array.
        
    Returns:
        np.ndarray: Image with batch dimension added.
    """
    return np.expand_dims(image, axis=0)