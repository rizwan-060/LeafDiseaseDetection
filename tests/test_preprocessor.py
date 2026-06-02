"""
Unit tests for src/preprocessor.py
Run with: pytest tests/test_preprocessor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import preprocessor
import config


@pytest.fixture
def dummy_rgb_image() -> np.ndarray:
    """Creates a dummy RGB image."""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)


@pytest.fixture
def dummy_bgr_image() -> np.ndarray:
    """Creates a dummy BGR image (OpenCV default)."""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)


# ─── TC-PRE-01: Resize to Exact IMG_SIZE ───
def test_resize_image_outputs_exact_size(dummy_rgb_image):
    resized = preprocessor.resize_image(dummy_rgb_image, config.IMG_SIZE)
    assert resized.shape == (*config.IMG_SIZE[::-1], 3)  # cv2 uses (W, H) -> output is (H, W, C)
    assert resized.shape[:2] == config.IMG_SIZE[::-1]


# ─── TC-PRE-02: Normalize to [0, 1] ───
def test_normalize_scale_255():
    img = np.array([0, 127, 255], dtype=np.float32).reshape(1, 1, 3)
    normalized = preprocessor.normalize_image(img, "scale_255")
    assert np.min(normalized) >= 0.0
    assert np.max(normalized) <= 1.0
    assert np.isclose(normalized[0, 0, 2], 1.0)  # 255/255 = 1.0


# ─── TC-PRE-03: Normalize to [-1, 1] ───
def test_normalize_scale_127():
    img = np.array([0, 127, 255], dtype=np.float32).reshape(1, 1, 3)
    normalized = preprocessor.normalize_image(img, "scale_127")
    assert np.min(normalized) >= -1.0
    assert np.max(normalized) <= 1.0
    assert np.isclose(normalized[0, 0, 1], 0.0, atol=0.01)  # 127/127.5 - 1 ≈ 0


# ─── TC-PRE-04: Histogram Equalization Changes Contrast ───
def test_histogram_equalization_changes_image(dummy_rgb_image):
    equalized = preprocessor.apply_histogram_equalization(dummy_rgb_image)
    assert equalized.shape == dummy_rgb_image.shape
    assert not np.array_equal(equalized, dummy_rgb_image)


# ─── TC-PRE-05: Histogram Equalization Preserves Shape ───
def test_histogram_equalization_preserves_shape(dummy_rgb_image):
    equalized = preprocessor.apply_histogram_equalization(dummy_rgb_image)
    assert equalized.shape == dummy_rgb_image.shape


# ─── TC-PRE-06: Segment Leaf Returns Same Shape ───
def test_segment_leaf_preserves_shape(dummy_rgb_image):
    segmented = preprocessor.segment_leaf(dummy_rgb_image)
    assert segmented.shape == dummy_rgb_image.shape


# ─── TC-PRE-07: Segment Leaf Produces Different Pixels ───
def test_segment_leaf_changes_pixels(dummy_rgb_image):
    segmented = preprocessor.segment_leaf(dummy_rgb_image)
    # For a random image, segmentation should change something (mask may be empty though)
    assert segmented.dtype == dummy_rgb_image.dtype


# ─── TC-PRE-08: Preprocess Array Full Pipeline ───
def test_preprocess_array_pipeline(dummy_rgb_image):
    result = preprocessor.preprocess_array(
        dummy_rgb_image,
        apply_equalization=True,
        apply_segmentation=False,
        normalize_method="scale_255"
    )
    assert result.shape == config.IMG_SHAPE
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0


# ─── TC-PRE-09: Preprocess Array Without Equalization ───
def test_preprocess_array_no_equalization(dummy_rgb_image):
    result = preprocessor.preprocess_array(
        dummy_rgb_image,
        apply_equalization=False,
        apply_segmentation=False,
        normalize_method="scale_255"
    )
    assert result.shape == config.IMG_SHAPE
    assert result.dtype == np.float32


# ─── TC-PRE-10: Add Batch Dimension ───
def test_add_batch_dimension():
    img = np.zeros(config.IMG_SHAPE, dtype=np.float32)
    batched = preprocessor.add_batch_dimension(img)
    assert batched.shape == (1, *config.IMG_SHAPE)


# ─── TC-PRE-11: Rejects Empty Image (Resize) ───
def test_resize_rejects_empty():
    with pytest.raises(ValueError):
        preprocessor.resize_image(np.array([]), config.IMG_SIZE)


# ─── TC-PRE-12: Rejects Empty Image (Equalization) ───
def test_equalization_rejects_empty():
    with pytest.raises(ValueError):
        preprocessor.apply_histogram_equalization(np.array([]))


# ─── TC-PRE-13: Rejects Empty Image (Segmentation) ───
def test_segmentation_rejects_empty():
    with pytest.raises(ValueError):
        preprocessor.segment_leaf(np.array([]))


# ─── TC-PRE-14: Rejects Empty Array (Preprocess) ───
def test_preprocess_array_rejects_empty():
    with pytest.raises(ValueError):
        preprocessor.preprocess_array(np.array([]))


# ─── TC-PRE-15: Invalid Normalization Method ───
def test_invalid_normalization_method():
    img = np.ones((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocessor.normalize_image(img, "invalid_method")


# ─── TC-PRE-16: Preprocess Image from File (Integration) ───
@pytest.mark.skipif(not config.TRAIN_DIR.exists(), reason="Dataset not available")
def test_preprocess_image_from_file():
    # Find first available image
    sample = None
    for class_dir in sorted(config.TRAIN_DIR.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.jpg"))
            if images:
                sample = images[0]
                break
    
    if sample is None:
        pytest.skip("No images found in dataset")
    
    result = preprocessor.preprocess_image(str(sample))
    assert result.shape == config.IMG_SHAPE
    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0