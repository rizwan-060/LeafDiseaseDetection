"""
Unit tests for src/data_loader.py
Run with: pytest tests/test_data_loader.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data_loader
import config
from tensorflow.keras.preprocessing.image import DirectoryIterator

# Skip data-dependent tests if dataset is not present
DATASET_AVAILABLE = config.TRAIN_DIR.exists() and config.VALID_DIR.exists()
TEST_DIR_AVAILABLE = config.TEST_DIR.exists()

@pytest.fixture(scope="module")
def train_gen():
    """Module-scoped train generator to avoid repeated directory scanning."""
    if not DATASET_AVAILABLE:
        pytest.skip("Dataset not found at expected path")
    return data_loader.load_train_generator()


@pytest.fixture(scope="module")
def val_gen():
    if not DATASET_AVAILABLE:
        pytest.skip("Dataset not found at expected path")
    return data_loader.load_validation_generator()


@pytest.fixture(scope="module")
def test_gen():
    if not TEST_DIR_AVAILABLE:
        pytest.skip("Test directory not found")
    return data_loader.load_test_generator()

# ─── TC-DL-01: Train Generator Type ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_load_train_generator_returns_directory_iterator(train_gen):
    """Training generator must be a DirectoryIterator."""
    assert isinstance(train_gen, DirectoryIterator)


# ─── TC-DL-02: Class Count ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_train_generator_has_38_classes(train_gen):
    """Dataset must have exactly 38 classes."""
    assert train_gen.num_classes == 38


# ─── TC-DL-03: Validation Batch Size ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_validation_generator_batch_size_matches_config(val_gen):
    """Validation batch size must match config.BATCH_SIZE."""
    assert val_gen.batch_size == config.BATCH_SIZE


# ─── TC-DL-04: Batch Image Shape ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_train_generator_yields_correct_batch_shape(train_gen):
    """Batch images must have shape (BATCH_SIZE, 224, 224, 3)."""
    batch_x, _ = next(train_gen)
    assert batch_x.shape[0] <= config.BATCH_SIZE
    assert batch_x.shape[1:] == config.IMG_SHAPE


# ─── TC-DL-05: Batch Label Shape ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_train_generator_yields_correct_label_shape(train_gen):
    """Batch labels must be one-hot encoded with shape (batch, 38)."""
    _, batch_y = next(train_gen)
    assert batch_y.shape[0] <= config.BATCH_SIZE
    assert batch_y.shape[1] == train_gen.num_classes


# ─── TC-DL-06: Class Indices Mapping ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_class_indices_mapping():
    """Class indices must be unique integers 0..37."""
    indices = data_loader.get_class_indices()
    assert isinstance(indices, dict)
    assert len(indices) == 38
    assert sorted(indices.values()) == list(range(38))


# ─── TC-DL-07: Train Generator Has Augmentation ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_train_generator_has_augmentation(train_gen):
    """Training generator's parent must have rotation and zoom enabled."""
    idg = train_gen.image_data_generator
    assert idg.rotation_range > 0
    assert idg.zoom_range != [1.0, 1.0]


# ─── TC-DL-08: Validation Generator Has No Augmentation ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_validation_generator_no_augmentation(val_gen):
    """Validation generator's parent must have no rotation or zoom."""
    idg = val_gen.image_data_generator
    assert idg.rotation_range == 0
    assert idg.zoom_range == [1.0, 1.0]


# ─── TC-DL-09: Image Count Positive ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_count_images_train_positive():
    """Training directory must contain images."""
    count = data_loader.count_images(config.TRAIN_DIR)
    assert count > 0
    assert count > 50000  # Expected ~70,295


# ─── TC-DL-10: get_num_classes Returns 38 ───
@pytest.mark.skipif(not DATASET_AVAILABLE, reason="Dataset not available")
def test_get_num_classes_returns_38():
    """get_num_classes must return 38 for the full dataset."""
    assert data_loader.get_num_classes() == 38


# ─── TC-DL-11: Missing Directory Raises Error (Train) ───
def test_load_train_generator_raises_on_missing_dir():
    """Must raise FileNotFoundError when training directory is missing."""
    original = config.TRAIN_DIR
    config.TRAIN_DIR = Path("nonexistent_path_12345")
    with pytest.raises(FileNotFoundError):
        data_loader.load_train_generator()
    config.TRAIN_DIR = original


# ─── TC-DL-12: Missing Directory Raises Error (Validation) ───
def test_load_validation_generator_raises_on_missing_dir():
    """Must raise FileNotFoundError when validation directory is missing."""
    original = config.VALID_DIR
    config.VALID_DIR = Path("nonexistent_path_12345")
    with pytest.raises(FileNotFoundError):
        data_loader.load_validation_generator()
    config.VALID_DIR = original


# ─── TC-DL-13: Test Generator Created ───
@pytest.mark.skipif(not TEST_DIR_AVAILABLE, reason="Test directory not available")
def test_test_generator_created(test_gen):
    """Test generator must instantiate successfully."""
    assert isinstance(test_gen, DirectoryIterator)
    assert test_gen.num_classes == 38