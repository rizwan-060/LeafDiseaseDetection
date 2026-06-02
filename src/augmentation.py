"""
Image Augmentation Pipeline for Leaf Disease Detection.
Implements rotation and zoom augmentation using TensorFlow/Keras
to prevent overfitting, as specified in the project guidelines.
"""

import os
from typing import Union

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def get_training_augmentation() -> ImageDataGenerator:
    """
    Creates an ImageDataGenerator with rotation and zoom augmentation
    for training data. This is the core pipeline component required by
    the project specification to prevent overfitting.
    
    Returns:
        ImageDataGenerator: Configured with rotation_range, zoom_range,
        width_shift, height_shift, and horizontal_flip.
    """
    return ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=config.AUG_ROTATION_RANGE,
        zoom_range=config.AUG_ZOOM_RANGE,
        width_shift_range=config.AUG_WIDTH_SHIFT,
        height_shift_range=config.AUG_HEIGHT_SHIFT,
        horizontal_flip=config.AUG_HORIZONTAL_FLIP,
        fill_mode=config.AUG_FILL_MODE,
    )


def get_validation_augmentation() -> ImageDataGenerator:
    """
    Creates an ImageDataGenerator for validation data.
    Only rescales — NO augmentation applied, to ensure valid metrics.
    
    Returns:
        ImageDataGenerator: Rescale only.
    """
    return ImageDataGenerator(rescale=1.0 / 255.0)


def get_test_augmentation() -> ImageDataGenerator:
    """
    Creates an ImageDataGenerator for test data.
    Only rescales — NO augmentation applied.
    
    Returns:
        ImageDataGenerator: Rescale only.
    """
    return ImageDataGenerator(rescale=1.0 / 255.0)


def apply_opencv_rotation(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Applies rotation to a single image using OpenCV.
    
    Args:
        image: Input image as numpy array (H, W, C) in RGB or BGR.
        angle: Rotation angle in degrees.
        
    Returns:
        np.ndarray: Rotated image with same dimensions as input.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (w, h), 
        borderMode=cv2.BORDER_REFLECT_101
    )
    return rotated


def apply_opencv_zoom(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Applies zoom to a single image using OpenCV.
    Zoom factor > 1.0 zooms in (crops center), < 1.0 zooms out.
    
    Args:
        image: Input image as numpy array (H, W, C).
        zoom_factor: Zoom multiplier.
        
    Returns:
        np.ndarray: Zoomed image resized back to original dimensions.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be positive")
    
    h, w = image.shape[:2]
    
    new_h = max(int(h / zoom_factor), 1)
    new_w = max(int(w / zoom_factor), 1)
    
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    y2 = y1 + new_h
    x2 = x1 + new_w
    
    cropped = image[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


def demonstrate_augmentation(
    image_path: Union[str, os.PathLike], 
    save_dir: Union[str, os.PathLike] = None
) -> dict:
    """
    Demonstrates augmentation on a single image for visualization.
    
    Args:
        image_path: Path to the image file.
        save_dir: Optional directory to save augmented samples.
        
    Returns:
        dict: Dictionary with keys 'original', 'rotated', 'zoomed', 'flipped'.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = {
        "original": image_rgb.copy(),
        "rotated": apply_opencv_rotation(image_rgb, config.AUG_ROTATION_RANGE),
        "zoomed": apply_opencv_zoom(image_rgb, 1.0 + config.AUG_ZOOM_RANGE),
    }
    
    if config.AUG_HORIZONTAL_FLIP:
        results["flipped"] = cv2.flip(image_rgb, 1)
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for name, img in results.items():
            save_path = os.path.join(save_dir, f"{name}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return results