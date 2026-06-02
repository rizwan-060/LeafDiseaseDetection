"""
Utility functions for the Leaf Disease Detection web application.
"""

import os
import uuid
from pathlib import Path
from typing import Union

import cv2
import numpy as np

import config


def save_uploaded_file(file_storage, upload_dir: Path = config.UPLOAD_DIR) -> Path:
    """
    Saves an uploaded file to disk with a unique filename.
    
    Args:
        file_storage: Flask FileStorage object.
        upload_dir: Directory to save the file.
        
    Returns:
        Path: Full path to the saved file.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename to prevent collisions
    ext = Path(file_storage.filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = upload_dir / unique_name
    
    file_storage.save(str(save_path))
    return save_path


def validate_image_file(filename: str) -> bool:
    """
    Checks if a filename has an allowed image extension.
    
    Args:
        filename: Name of the uploaded file.
        
    Returns:
        bool: True if valid image extension.
    """
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    ext = Path(filename).suffix.lower()
    return ext in allowed


def cleanup_uploads(upload_dir: Path = config.UPLOAD_DIR, max_files: int = 50):
    """
    Removes oldest uploaded files if directory exceeds max_files.
    Prevents disk bloat from repeated uploads.
    
    Args:
        upload_dir: Directory to clean.
        max_files: Maximum number of files to keep.
    """
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        return
    
    files = sorted(upload_dir.iterdir(), key=lambda f: f.stat().st_mtime)
    while len(files) > max_files:
        oldest = files.pop(0)
        if oldest.is_file():
            oldest.unlink()


def encode_image_base64(image_path: Union[str, Path]) -> str:
    """
    Encodes an image file to base64 string for inline HTML display.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        str: Base64-encoded image string with data URI prefix.
    """
    import base64
    
    path = Path(image_path)
    if not path.exists():
        return ""
    
    ext = path.suffix.lower().replace('.', '')
    if ext == 'jpg':
        ext = 'jpeg'
    
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:image/{ext};base64,{data}"


def get_health_status() -> dict:
    """
    Returns system health status for the /health endpoint.
    
    Returns:
        dict: Status information.
    """
    model_exists = config.MODEL_PATH.exists()
    data_exists = config.TRAIN_DIR.exists()
    
    return {
        "status": "ok" if model_exists else "degraded",
        "model_loaded": model_exists,
        "dataset_available": data_exists,
        "model_path": str(config.MODEL_PATH),
        "num_classes": len(config.CLASS_NAMES) if config.CLASS_NAMES else 0
    }