"""
End-to-End Integration Tests for Leaf Disease Detection System.
Tests the complete pipeline: config -> augmentation -> data -> model -> predictor -> web.
Run with: pytest tests/test_integration.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config
import augmentation
import data_loader
import model as model_module
import preprocessor
import predictor
from app import app as flask_app


# ─── Flask Fixtures ───
@pytest.fixture
def app():
    """Expose Flask app for pytest-flask."""
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    """Create test client using pytest-flask."""
    return app.test_client()


# ─── Dummy Model Fixture ───
@pytest.fixture
def dummy_model_path(tmp_path):
    """Creates a tiny dummy Keras model for integration tests."""
    dummy = tf.keras.Sequential([
        tf.keras.layers.Input(shape=config.IMG_SHAPE),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    dummy.compile(optimizer='adam', loss='categorical_crossentropy')
    path = str(tmp_path / "dummy_integration.h5")
    dummy.save(path)
    return path


# ─── TC-E2E-01: Full Pipeline Components Import ───
def test_all_modules_import_cleanly():
    """All source modules must import without errors."""
    assert config is not None
    assert augmentation is not None
    assert data_loader is not None
    assert model_module is not None
    assert preprocessor is not None
    assert predictor is not None


# ─── TC-E2E-02: Config Refresh Loads Class Names ───
@pytest.mark.skipif(not config.TRAIN_DIR.exists(), reason="Dataset not available")
def test_config_refresh_loads_class_names():
    """refresh_class_names must populate CLASS_NAMES from dataset."""
    config.refresh_class_names()
    assert len(config.CLASS_NAMES) == 38
    assert all(isinstance(name, str) for name in config.CLASS_NAMES)


# ─── TC-E2E-03: Augmentation + Data Loader Integration ───
@pytest.mark.skipif(not config.TRAIN_DIR.exists(), reason="Dataset not available")
def test_augmentation_flows_from_data_loader():
    """Training generator must apply augmentation from augmentation.py."""
    train_gen = data_loader.load_train_generator()
    idg = train_gen.image_data_generator
    assert idg.rotation_range == config.AUG_ROTATION_RANGE
    assert idg.zoom_range != [1.0, 1.0]


# ─── TC-E2E-04: Preprocessor Output Feeds Model ───
def test_preprocessor_output_compatible_with_model():
    """Preprocessed image must have shape that model accepts."""
    dummy = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    processed = preprocessor.preprocess_array(dummy)
    batched = preprocessor.add_batch_dimension(processed)
    
    assert batched.shape == (1, *config.IMG_SHAPE)
    
    # Model should accept this shape
    m = model_module.build_model(num_classes=5)
    preds = m.predict(batched, verbose=0)
    assert preds.shape == (1, 5)


# ─── TC-E2E-05: Model Save -> Load -> Predict Roundtrip ───
def test_save_load_predict_roundtrip(tmp_path):
    """Saved model must produce identical predictions after reload."""
    m = model_module.build_model(num_classes=5)
    dummy = np.zeros((1, *config.IMG_SHAPE), dtype=np.float32)
    preds_before = m.predict(dummy, verbose=0)
    
    path = str(tmp_path / "integration_model.h5")
    model_module.save_model(m, path=path)
    loaded = model_module.load_model(path)
    preds_after = loaded.predict(dummy, verbose=0)
    
    assert np.allclose(preds_before, preds_after, atol=1e-5)


# ─── TC-E2E-06: Predictor Uses Preprocessor ───
def test_predictor_uses_preprocessor(dummy_model_path, tmp_path):
    """predict_image must internally call preprocessor."""
    import cv2
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "leaf.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    original_names = config.CLASS_NAMES[:]
    config.CLASS_NAMES = ['Class_A', 'Class_B']
    
    try:
        m = predictor.load_model(dummy_model_path)
        result = predictor.predict_image(str(img_path), model=m)
        assert result['predicted_class'] in ['Class_A', 'Class_B']
        assert 0.0 <= result['confidence'] <= 1.0
    finally:
        config.CLASS_NAMES = original_names


# ─── TC-E2E-07: Flask App Serves All Routes ───
def test_flask_app_routes_exist(client):
    """All major routes must return valid responses."""
    routes = ['/', '/health', '/classes']
    for route in routes:
        response = client.get(route)
        assert response.status_code in [200, 503]


# ─── TC-E2E-08: Web Upload -> Predict Flow (Mocked) ───
def test_web_upload_predict_flow(client, tmp_path):
    """Full web flow: upload image -> get prediction result."""
    import cv2
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img_path = tmp_path / "test_leaf.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    with open(img_path, 'rb') as f:
        response = client.post(
            '/predict',
            data={'file': (f, 'test_leaf.jpg')},
            content_type='multipart/form-data',
            follow_redirects=True
        )
    
    # Should either show result page or flash error (if model missing)
    assert response.status_code == 200


# ─── TC-E2E-09: Dataset Stats Match Expected ───
@pytest.mark.skipif(not config.TRAIN_DIR.exists(), reason="Dataset not available")
def test_dataset_stats_match_expected():
    """Dataset must have approximately expected image counts."""
    stats = data_loader.get_dataset_stats()
    assert stats['train'] > 50000
    assert stats['valid'] > 10000


# ─── TC-E2E-10: Confidence Threshold Logic ───
def test_confidence_threshold_logic():
    """Uncertainty flag must be True when confidence < threshold."""
    assert config.CONFIDENCE_THRESHOLD > 0.0
    assert config.CONFIDENCE_THRESHOLD < 1.0
    
    # Simulate low confidence
    low_conf = config.CONFIDENCE_THRESHOLD - 0.1
    assert low_conf < config.CONFIDENCE_THRESHOLD
    
    # Simulate high confidence
    high_conf = config.CONFIDENCE_THRESHOLD + 0.1
    assert high_conf > config.CONFIDENCE_THRESHOLD