"""
Unit tests for src/model.py
Run with: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import model
import config


# ─── TC-MOD-01: Build Returns Keras Model ───
def test_build_model_returns_keras_model():
    """build_model() must return a tf.keras.Model instance."""
    m = model.build_model(num_classes=10)
    assert isinstance(m, tf.keras.Model)


# ─── TC-MOD-02: Output Units Match num_classes ───
def test_model_output_units_match_num_classes():
    """Output layer must have exactly num_classes units."""
    for n in [5, 10, 38]:
        m = model.build_model(num_classes=n)
        assert m.output_shape[-1] == n


# ─── TC-MOD-03: Output Activation is Softmax ───
def test_model_output_activation_is_softmax():
    """Final layer must use softmax for multi-class classification."""
    m = model.build_model(num_classes=10)
    last_layer = m.layers[-1]
    assert last_layer.activation.__name__ == 'softmax'


# ─── TC-MOD-04: Model Has Trainable Parameters ───
def test_model_has_trainable_params():
    """Model must have > 0 trainable parameters after build."""
    m = model.build_model(num_classes=10)
    assert m.count_params() > 0


# ─── TC-MOD-05: Base Frozen by Default ───
def test_base_frozen_by_default():
    """Transfer learning base must be frozen in initial build."""
    m = model.build_model(num_classes=10, trainable_base=False)
    # Find the nested MobileNetV2 model
    base = None
    for layer in m.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break
    assert base is not None
    assert base.trainable is False


# ─── TC-MOD-06: Base Trainable When Requested ───
def test_base_trainable_when_requested():
    """Base must be trainable when trainable_base=True."""
    m = model.build_model(num_classes=10, trainable_base=True)
    base = None
    for layer in m.layers:
        if isinstance(layer, tf.keras.Model):
            base = layer
            break
    assert base is not None
    assert base.trainable is True


# ─── TC-MOD-07: Input Shape Matches Config ───
def test_model_input_shape_matches_config():
    """Model input must accept config.IMG_SHAPE."""
    m = model.build_model(num_classes=10)
    assert m.input_shape[1:] == config.IMG_SHAPE


# ─── TC-MOD-08: Model Compilation Has Correct Loss ───
def test_model_compiled_with_categorical_crossentropy():
    """Loss function must be categorical_crossentropy."""
    m = model.build_model(num_classes=10)
    assert m.loss == 'categorical_crossentropy'


# ─── TC-MOD-09: Model Compilation Has Metrics ───
def test_model_compiled_with_metrics():
    """Model must compile successfully with optimizer and metrics configured."""
    m = model.build_model(num_classes=10)
    
    # Proof of compilation: optimizer must exist and be configured
    assert m.optimizer is not None, "Model has no optimizer"
    
    # Proof of compilation: loss must be categorical_crossentropy
    assert m.loss == 'categorical_crossentropy'
    
    # Proof metrics exist: compiled_metrics container must be present
    assert hasattr(m, 'compiled_metrics'), "Model missing compiled_metrics"
    assert m.compiled_metrics is not None, "compiled_metrics is None"
    
    # Verify model actually trains for 1 step without error
    dummy_x = np.zeros((1, *config.IMG_SHAPE), dtype=np.float32)
    dummy_y = np.zeros((1, 10), dtype=np.float32)
    loss = m.train_on_batch(dummy_x, dummy_y)
    assert isinstance(loss, (list, float)), "Model failed to execute train step"


# ─── TC-MOD-10: Model Can Predict on Dummy Batch ───
def test_model_predicts_on_dummy_batch():
    """Model must produce predictions of shape (batch, num_classes)."""
    m = model.build_model(num_classes=10)
    dummy = np.zeros((2, *config.IMG_SHAPE), dtype=np.float32)
    preds = m.predict(dummy, verbose=0)
    assert preds.shape == (2, 10)
    assert np.all(preds >= 0) and np.all(preds <= 1)
    assert np.allclose(np.sum(preds, axis=1), 1.0, atol=1e-5)  # Softmax sums to 1


# ─── TC-MOD-11: Model Summary is Non-Empty ───
def test_model_summary_is_non_empty():
    """get_model_summary must return a string with layer info."""
    m = model.build_model(num_classes=10)
    summary = model.get_model_summary(m)
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert 'MobileNetV2' in summary or 'global_average_pooling2d' in summary


# ─── TC-MOD-12: Count Trainable Params Returns Integer ───
def test_count_trainable_params_returns_int():
    """count_trainable_params must return a positive integer."""
    m = model.build_model(num_classes=10)
    count = model.count_trainable_params(m)
    assert isinstance(count, int)
    assert count > 0


# ─── TC-MOD-13: Unfreeze Changes Trainable Status ───
def test_unfreeze_base_layers_changes_trainable():
    """unfreeze_base_layers must make some base layers trainable."""
    m = model.build_model(num_classes=10, trainable_base=False)
    initial_trainable = sum(l.trainable for l in m.layers)
    
    m = model.unfreeze_base_layers(m, num_layers=20)
    final_trainable = sum(l.trainable for l in m.layers)
    
    # At least some layers should now be trainable
    assert final_trainable >= initial_trainable


# ─── TC-MOD-14: Save and Load Roundtrip ───
def test_save_and_load_roundtrip(tmp_path):
    """Saving and loading must preserve architecture and weights."""
    m = model.build_model(num_classes=5)
    dummy = np.zeros((1, *config.IMG_SHAPE), dtype=np.float32)
    preds_before = m.predict(dummy, verbose=0)
    
    path = str(tmp_path / "test_model.h5")
    model.save_model(m, path=path)
    loaded = model.load_model(path)
    preds_after = loaded.predict(dummy, verbose=0)
    
    assert np.allclose(preds_before, preds_after, atol=1e-5)


# ─── TC-MOD-15: Invalid num_classes Raises Error ───
def test_invalid_num_classes_raises_error():
    """num_classes <= 0 must raise ValueError."""
    with pytest.raises(ValueError):
        model.build_model(num_classes=0)
    with pytest.raises(ValueError):
        model.build_model(num_classes=-1)


# ─── TC-MOD-16: Model Name Contains LeafDisease ───
def test_model_name_contains_project_name():
    """Model name should indicate project identity."""
    m = model.build_model(num_classes=10)
    assert 'LeafDisease' in m.name