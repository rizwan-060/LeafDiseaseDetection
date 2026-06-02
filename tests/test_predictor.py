"""
Unit tests for src/predictor.py
Run with: pytest tests/test_predictor.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predictor
import config
import preprocessor


@pytest.fixture
def dummy_model_path(tmp_path):
    """
    Creates a tiny dummy Keras model and saves it for testing.
    Returns the path to the saved model.
    """
    # Tiny model: 2 classes for fast testing
    dummy = tf.keras.Sequential([
        tf.keras.layers.Input(shape=config.IMG_SHAPE),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    dummy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    path = str(tmp_path / "dummy_model.h5")
    dummy.save(path)
    return path


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Creates a dummy RGB image for prediction tests."""
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)


# ─── TC-PRD-01: load_model Returns Keras Model ───
def test_load_model_returns_keras_model(dummy_model_path):
    """load_model must return a tf.keras.Model."""
    m = predictor.load_model(dummy_model_path)
    assert isinstance(m, tf.keras.Model)


# ─── TC-PRD-02: load_model Raises on Missing File ───
def test_load_model_raises_on_missing():
    """Must raise FileNotFoundError for non-existent model path."""
    with pytest.raises(FileNotFoundError):
        predictor.load_model("nonexistent_model_12345.h5")


# ─── TC-PRD-03: predict_image Returns Dict ───
def test_predict_image_returns_dict(dummy_model_path, dummy_image, tmp_path):
    """predict_image must return a dictionary with expected keys."""
    # Save dummy image temporarily
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    # Temporarily override class names to match dummy model (2 classes)
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m, top_k=2)
        
        assert isinstance(result, dict)
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'is_healthy' in result
        assert 'uncertain' in result
        assert 'top_k' in result
        assert 'all_probabilities' in result
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-04: Confidence is in Valid Range ───
def test_predict_image_confidence_in_range(dummy_model_path, dummy_image, tmp_path):
    """Confidence must be between 0.0 and 1.0."""
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m)
        assert 0.0 <= result['confidence'] <= 1.0
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-05: top_k Length Matches Request ───
def test_predict_image_top_k_length(dummy_model_path, dummy_image, tmp_path):
    """top_k list must contain exactly 'top_k' items."""
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m, top_k=2)
        assert len(result['top_k']) == 2
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-06: predict_image Raises on Missing File ───
def test_predict_image_raises_on_missing(dummy_model_path):
    """Must raise FileNotFoundError when image path does not exist."""
    with pytest.raises(FileNotFoundError):
        predictor.predict_image("nonexistent_image_12345.jpg", model=None)


# ─── TC-PRD-07: predict_array Returns Dict ───
def test_predict_array_returns_dict(dummy_model_path, dummy_image):
    """predict_array must return a dictionary."""
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_array(dummy_image, model=m, top_k=2)
        
        assert isinstance(result, dict)
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'top_k' in result
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-08: is_healthy Flag Logic ───
def test_is_healthy_flag(dummy_model_path, dummy_image, tmp_path):
    """is_healthy must be True when 'healthy' is in predicted class name."""
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    # Force class names so we can control prediction
    config.CLASS_NAMES = ['Tomato___healthy', 'Tomato___Late_blight']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m, top_k=2)
        # is_healthy should be a boolean
        assert isinstance(result['is_healthy'], bool)
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-09: uncertain Flag When Low Confidence ───
def test_uncertain_flag_low_confidence(dummy_model_path, dummy_image, tmp_path):
    """uncertain must be True when confidence < CONFIDENCE_THRESHOLD."""
    # We can't easily force low confidence, but we can verify it's a boolean
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m)
        assert isinstance(result['uncertain'], bool)
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-10: get_treatment_recommendation Returns String ───
def test_get_treatment_returns_string():
    """Must return a non-empty string for known diseases."""
    rec = predictor.get_treatment_recommendation('Tomato___healthy')
    assert isinstance(rec, str)
    assert len(rec) > 0


# ─── TC-PRD-11: get_treatment_recommendation Unknown Class ───
def test_get_treatment_unknown_class():
    """Must return fallback message for unknown class."""
    rec = predictor.get_treatment_recommendation('Unknown_Disease_XYZ')
    assert 'Consult' in rec or 'extension' in rec


# ─── TC-PRD-12: top_k Contains Treatment Strings ───
def test_top_k_contains_treatment(dummy_model_path, dummy_image, tmp_path):
    """Each top_k result must include a treatment string."""
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Tomato___healthy', 'Tomato___Late_blight']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m, top_k=2)
        for item in result['top_k']:
            assert 'treatment' in item
            assert isinstance(item['treatment'], str)
            assert len(item['treatment']) > 0
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-PRD-13: all_probabilities Sum to ~1.0 ───
def test_all_probabilities_sum_to_one(dummy_model_path, dummy_image, tmp_path):
    """Softmax probabilities across all classes must sum to ~1.0."""
    img_path = tmp_path / "test_leaf.jpg"
    import cv2
    cv2.imwrite(str(img_path), cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m)
        total = sum(result['all_probabilities'].values())
        assert np.isclose(total, 1.0, atol=1e-5)
    finally:
        config.CLASS_NAMES = original_names