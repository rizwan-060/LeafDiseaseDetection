"""
Unit tests for src/config.py
Run with: pytest tests/test_config.py -v
"""

import os
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import config


# ─── Path Tests ───
def test_base_dir_exists():
    """TC-CFG-01: BASE_DIR must exist and be a directory."""
    assert config.BASE_DIR.exists()
    assert config.BASE_DIR.is_dir()


def test_data_dir_path_type():
    """TC-CFG-02: DATA_DIR must be a Path object."""
    assert isinstance(config.DATA_DIR, Path)


def test_model_dir_auto_created():
    """TC-CFG-03: MODEL_DIR must exist (auto-created by config)."""
    assert config.MODEL_DIR.exists()
    assert config.MODEL_DIR.is_dir()


def test_upload_dir_auto_created():
    """TC-CFG-04: UPLOAD_DIR must exist (auto-created by config)."""
    assert config.UPLOAD_DIR.exists()
    assert config.UPLOAD_DIR.is_dir()


def test_model_path_extension():
    """TC-CFG-05: MODEL_PATH must end with .h5 or .keras."""
    assert str(config.MODEL_PATH).endswith((".h5", ".keras"))


# ─── Image Parameter Tests ───
def test_img_size_is_tuple_of_positive_ints():
    """TC-CFG-06: IMG_SIZE must be a tuple of two positive integers."""
    assert isinstance(config.IMG_SIZE, tuple)
    assert len(config.IMG_SIZE) == 2
    assert all(isinstance(x, int) and x > 0 for x in config.IMG_SIZE)


def test_img_channels_positive():
    """TC-CFG-07: IMG_CHANNELS must be a positive integer."""
    assert isinstance(config.IMG_CHANNELS, int)
    assert config.IMG_CHANNELS > 0


def test_img_shape_consistency():
    """TC-CFG-08: IMG_SHAPE must match (W, H, C)."""
    expected = (*config.IMG_SIZE, config.IMG_CHANNELS)
    assert config.IMG_SHAPE == expected


# ─── Hyperparameter Tests ───
def test_batch_size_positive_int():
    """TC-CFG-09: BATCH_SIZE must be a positive integer."""
    assert isinstance(config.BATCH_SIZE, int)
    assert config.BATCH_SIZE > 0


def test_epochs_positive_int():
    """TC-CFG-10: EPOCHS must be a positive integer."""
    assert isinstance(config.EPOCHS, int)
    assert config.EPOCHS > 0


def test_learning_rate_positive_float():
    """TC-CFG-11: LEARNING_RATE must be a positive float."""
    assert isinstance(config.LEARNING_RATE, float)
    assert config.LEARNING_RATE > 0.0


# ─── Augmentation Tests (CRITICAL: Matches project guideline) ───
def test_augmentation_rotation_range_non_negative():
    """TC-AUG-01: Rotation range must be >= 0."""
    assert isinstance(config.AUG_ROTATION_RANGE, int)
    assert config.AUG_ROTATION_RANGE >= 0


def test_augmentation_zoom_range_valid():
    """TC-AUG-02: Zoom range must be a non-negative float."""
    assert isinstance(config.AUG_ZOOM_RANGE, float)
    assert config.AUG_ZOOM_RANGE >= 0.0


def test_augmentation_fill_mode_valid():
    """TC-AUG-03: Fill mode must be one of standard OpenCV/Keras modes."""
    valid_modes = {"constant", "nearest", "reflect", "wrap"}
    assert config.AUG_FILL_MODE in valid_modes


def test_horizontal_flip_is_boolean():
    """TC-AUG-04: Horizontal flip flag must be a boolean."""
    assert isinstance(config.AUG_HORIZONTAL_FLIP, bool)


# ─── Inference Tests ───
def test_confidence_threshold_between_zero_and_one():
    """TC-CFG-12: CONFIDENCE_THRESHOLD must be in (0, 1)."""
    assert 0.0 < config.CONFIDENCE_THRESHOLD < 1.0


def test_top_k_positive_int():
    """TC-CFG-13: TOP_K_PREDICTIONS must be a positive integer."""
    assert isinstance(config.TOP_K_PREDICTIONS, int)
    assert config.TOP_K_PREDICTIONS > 0


# ─── Dynamic Class Loading Tests ───
def test_refresh_class_names_is_callable():
    """TC-CFG-14: refresh_class_names must be a callable function."""
    assert callable(config.refresh_class_names)


def test_get_class_names_from_directory_handles_missing_dir():
    """TC-CFG-15: Must return empty list for non-existent directory."""
    result = config.get_class_names_from_directory(Path("nonexistent_folder_xyz"))
    assert result == []


def test_class_names_is_list_of_strings():
    """TC-CFG-16: CLASS_NAMES must be a list of strings (may be empty before data download)."""
    assert isinstance(config.CLASS_NAMES, list)
    assert all(isinstance(name, str) for name in config.CLASS_NAMES)