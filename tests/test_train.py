"""
Unit tests for src/train.py
Run with: pytest tests/test_train.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train
import config


# ─── TC-TRN-01: Module Imports Successfully ───
def test_train_module_imports():
    """train.py must import without errors."""
    assert train is not None


# ─── TC-TRN-02: get_callbacks Returns List ───
def test_get_callbacks_returns_list():
    """get_callbacks must return a non-empty list."""
    callbacks = train.get_callbacks()
    assert isinstance(callbacks, list)
    assert len(callbacks) > 0


# ─── TC-TRN-03: Callbacks Are Correct Types ───
def test_callbacks_are_correct_types():
    """Callbacks must include ModelCheckpoint, EarlyStopping, ReduceLROnPlateau."""
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    
    callbacks = train.get_callbacks()
    types = [type(c) for c in callbacks]
    
    assert any(issubclass(t, ModelCheckpoint) for t in types)
    assert any(issubclass(t, EarlyStopping) for t in types)
    assert any(issubclass(t, ReduceLROnPlateau) for t in types)


# ─── TC-TRN-04: ModelCheckpoint Saves to Correct Path ───
def test_checkpoint_saves_to_config_path():
    """ModelCheckpoint must target config.MODEL_PATH."""
    callbacks = train.get_callbacks()
    checkpoint = [c for c in callbacks if type(c).__name__ == 'ModelCheckpoint'][0]
    assert str(config.MODEL_PATH) in checkpoint.filepath


# ─── TC-TRN-05: EarlyStopping Monitors Val Accuracy ───
def test_early_stopping_monitors_val_accuracy():
    """EarlyStopping must monitor val_accuracy."""
    callbacks = train.get_callbacks()
    early_stop = [c for c in callbacks if type(c).__name__ == 'EarlyStopping'][0]
    assert early_stop.monitor == 'val_accuracy'


# ─── TC-TRN-06: ReduceLROnPlateau Monitors Val Loss ───
def test_lr_reduce_monitors_val_loss():
    """ReduceLROnPlateau must monitor val_loss."""
    callbacks = train.get_callbacks()
    lr_reduce = [c for c in callbacks if type(c).__name__ == 'ReduceLROnPlateau'][0]
    assert lr_reduce.monitor == 'val_loss'


# ─── TC-TRN-07: plot_training_history Creates File ───
def test_plot_training_history_creates_file(tmp_path):
    """plot_training_history must save a PNG file."""
    dummy_history = {
        'accuracy': [0.1, 0.5, 0.9],
        'val_accuracy': [0.1, 0.4, 0.8],
        'loss': [2.0, 1.0, 0.3],
        'val_loss': [2.1, 1.1, 0.4]
    }
    
    # Mock history object
    class MockHistory:
        history = dummy_history
    
    plot_path = train.plot_training_history(MockHistory(), save_dir=tmp_path)
    assert Path(plot_path).exists()
    assert Path(plot_path).suffix == '.png'


# ─── TC-TRN-08: plot_training_history Handles Dict Input ───
def test_plot_training_history_handles_dict(tmp_path):
    """plot_training_history must accept a plain dict."""
    dummy_history = {
        'accuracy': [0.2, 0.6],
        'val_accuracy': [0.2, 0.5],
        'loss': [1.5, 0.8],
        'val_loss': [1.6, 0.9]
    }
    
    plot_path = train.plot_training_history(dummy_history, save_dir=tmp_path)
    assert Path(plot_path).exists()


# ─── TC-TRN-09: train Function Exists and is Callable ───
def test_train_function_exists():
    """train() function must exist and be callable."""
    assert callable(train.train)


# ─── TC-TRN-10: train Function Accepts Keyword Args ───
def test_train_function_signature():
    """train() must accept epochs, batch_size, learning_rate, fine_tune."""
    import inspect
    sig = inspect.signature(train.train)
    params = list(sig.parameters.keys())
    assert 'epochs' in params
    assert 'batch_size' in params
    assert 'learning_rate' in params
    assert 'fine_tune' in params