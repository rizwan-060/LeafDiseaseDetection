"""
Unit tests for src/augmentation.py
Run with: pytest tests/test_augmentation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import augmentation
import config


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Creates a dummy RGB image for testing."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


# ─── TC-AUG-01: Training Generator Created ───
def test_training_generator_is_created():
    gen = augmentation.get_training_augmentation()
    assert gen is not None
    assert hasattr(gen, "flow")

# ─── TC-AUG-02: Validation Generator Created (No Augmentation) ───
def test_validation_generator_is_created():
    """Validation generator must instantiate with no augmentation."""
    gen = augmentation.get_validation_augmentation()
    assert gen is not None
    assert gen.rotation_range == 0
    # Keras stores zoom_range as [1.0, 1.0] when disabled (no zoom = scale 1.0)
    assert gen.zoom_range == [1.0, 1.0] or gen.zoom_range == 0
    assert gen.horizontal_flip is False


# ─── TC-AUG-03: Training Generator Has Rotation ───
def test_training_generator_has_rotation():
    gen = augmentation.get_training_augmentation()
    assert gen.rotation_range == config.AUG_ROTATION_RANGE
    assert gen.rotation_range > 0


# ─── TC-AUG-04: Training Generator Has Zoom ───
def test_training_generator_has_zoom():
    """Training generator must have zoom_range matching config."""
    gen = augmentation.get_training_augmentation()
    zoom = gen.zoom_range
    # Keras internally converts zoom_range=0.2 into [0.8, 1.2]
    if isinstance(zoom, (list, tuple)):
        assert zoom[0] < zoom[1]  # Lower bound < upper bound
        assert zoom != [1.0, 1.0]  # Must actually apply zoom
    else:
        assert zoom > 0


# ─── TC-AUG-05: Pipeline Includes Both Rotation AND Zoom ───
def test_training_generator_has_both_rotation_and_zoom():
    gen = augmentation.get_training_augmentation()
    assert gen.rotation_range > 0
    assert gen.zoom_range is not None
    assert gen.zoom_range != 0


# ─── TC-AUG-06: Horizontal Flip Toggleable ───
def test_horizontal_flip_toggleable():
    gen = augmentation.get_training_augmentation()
    assert gen.horizontal_flip == config.AUG_HORIZONTAL_FLIP


# ─── TC-AUG-07: Rescaling Applied ───
def test_training_generator_rescales():
    gen = augmentation.get_training_augmentation()
    assert gen.rescale == 1.0 / 255.0


# ─── TC-AUG-08: OpenCV Rotation Preserves Shape ───
def test_opencv_rotation_preserves_shape(dummy_image):
    rotated = augmentation.apply_opencv_rotation(dummy_image, 45)
    assert rotated.shape == dummy_image.shape


# ─── TC-AUG-09: OpenCV Rotation Actually Changes Pixels ───
def test_opencv_rotation_changes_pixels(dummy_image):
    rotated = augmentation.apply_opencv_rotation(dummy_image, 30)
    assert not np.array_equal(rotated, dummy_image)


# ─── TC-AUG-10: OpenCV Zoom Preserves Shape ───
def test_opencv_zoom_preserves_shape(dummy_image):
    zoomed = augmentation.apply_opencv_zoom(dummy_image, 1.2)
    assert zoomed.shape == dummy_image.shape


# ─── TC-AUG-11: OpenCV Zoom Actually Changes Pixels ───
def test_opencv_zoom_changes_pixels(dummy_image):
    zoomed = augmentation.apply_opencv_zoom(dummy_image, 1.2)
    assert not np.array_equal(zoomed, dummy_image)


# ─── TC-AUG-12: OpenCV Zoom Out Preserves Shape ───
def test_opencv_zoom_out_preserves_shape(dummy_image):
    zoomed = augmentation.apply_opencv_zoom(dummy_image, 0.8)
    assert zoomed.shape == dummy_image.shape


# ─── TC-AUG-13: OpenCV Rotation Preserves Aspect Ratio ───
def test_opencv_rotation_preserves_aspect_ratio(dummy_image):
    rotated = augmentation.apply_opencv_rotation(dummy_image, 30)
    original_ratio = dummy_image.shape[0] / dummy_image.shape[1]
    rotated_ratio = rotated.shape[0] / rotated.shape[1]
    assert abs(original_ratio - rotated_ratio) < 0.01


# ─── TC-AUG-14: OpenCV Rejects Empty Image (Rotation) ───
def test_opencv_rotation_rejects_empty():
    with pytest.raises(ValueError):
        augmentation.apply_opencv_rotation(np.array([]), 45)


# ─── TC-AUG-15: OpenCV Rejects Empty Image (Zoom) ───
def test_opencv_zoom_rejects_empty():
    with pytest.raises(ValueError):
        augmentation.apply_opencv_zoom(np.array([]), 1.2)


# ─── TC-AUG-16: OpenCV Rejects Invalid Zoom Factor ───
def test_opencv_zoom_rejects_invalid_factor(dummy_image):
    with pytest.raises(ValueError):
        augmentation.apply_opencv_zoom(dummy_image, 0.0)


# ─── TC-AUG-17: Dynamic Augmentation Produces Variation ───
def test_dynamic_augmentation_produces_variation(dummy_image):
    gen = augmentation.get_training_augmentation()
    img_batch = np.expand_dims(dummy_image, axis=0)
    flow = gen.flow(img_batch, batch_size=1, seed=None)
    batch1 = next(flow)[0]
    batch2 = next(flow)[0]
    assert not np.array_equal(batch1, batch2)


# ─── TC-AUG-18: Augmented Pixel Values in Valid Range ───
def test_augmented_pixel_values_in_range(dummy_image):
    gen = augmentation.get_training_augmentation()
    img_batch = np.expand_dims(dummy_image, axis=0)
    flow = gen.flow(img_batch, batch_size=1, seed=42)
    augmented = next(flow)[0]
    assert np.min(augmented) >= 0.0
    assert np.max(augmented) <= 1.0


# ─── TC-AUG-19: Demonstration Raises on Missing File ───
def test_demonstrate_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        augmentation.demonstrate_augmentation("nonexistent_image_xyz.jpg")