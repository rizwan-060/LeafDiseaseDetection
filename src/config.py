"""
Configuration module for Leaf Disease Detection System.
All hyperparameters, paths, and constants are centralized here.
"""

import os
from pathlib import Path
from typing import Tuple, List


# ─── Base Paths ───
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Dataset is nested inside a folder with spaces in the name
DATASET_FOLDER_NAME: str = "New Plant Diseases Dataset(Augmented)"
DATA_DIR: Path = BASE_DIR / "data" / DATASET_FOLDER_NAME
TRAIN_DIR: Path = DATA_DIR / "train"
VALID_DIR: Path = DATA_DIR / "valid"
TEST_DIR: Path = DATA_DIR / "test"

MODEL_DIR: Path = BASE_DIR / "models"
MODEL_PATH: Path = MODEL_DIR / "leaf_disease_model.h5"
TFLITE_PATH: Path = MODEL_DIR / "leaf_disease_model.tflite"

UPLOAD_DIR: Path = BASE_DIR / "uploads"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─── Image Parameters ───
IMG_SIZE: Tuple[int, int] = (224, 224)  # Width, Height
IMG_CHANNELS: int = 3
IMG_SHAPE: Tuple[int, int, int] = (*IMG_SIZE, IMG_CHANNELS)


# ─── Training Hyperparameters ───
BATCH_SIZE: int = 32
EPOCHS: int = 20
LEARNING_RATE: float = 1e-4
SEED: int = 42


# ─── Augmentation Parameters (Project Requirement) ───
AUG_ROTATION_RANGE: int = 40          # Rotation in degrees
AUG_ZOOM_RANGE: float = 0.2           # Zoom: 1 ± 0.2 (i.e., 0.8× to 1.2×)
AUG_WIDTH_SHIFT: float = 0.2
AUG_HEIGHT_SHIFT: float = 0.2
AUG_HORIZONTAL_FLIP: bool = True
AUG_FILL_MODE: str = "nearest"


# ─── Model Parameters ───
NUM_CLASSES_FALLBACK: int = 38


# ─── Inference Parameters ───
CONFIDENCE_THRESHOLD: float = 0.60
TOP_K_PREDICTIONS: int = 3


# ─── Class Names ───
CLASS_NAMES: List[str] = []


def get_class_names_from_directory(directory: Path) -> List[str]:
    """
    Dynamically reads class names from dataset subdirectories.
    Returns sorted list of folder names.
    """
    if not directory.exists():
        return []
    return sorted([
        d.name for d in directory.iterdir() 
        if d.is_dir() and not d.name.startswith(".")
    ])


def refresh_class_names() -> None:
    """Refresh CLASS_NAMES from the training directory."""
    global CLASS_NAMES
    CLASS_NAMES = get_class_names_from_directory(TRAIN_DIR)


# Initialize on import (will be empty if data/ not yet present)
refresh_class_names()