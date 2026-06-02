"""
Unit tests for src/evaluate.py
Run with: pytest tests/test_evaluate.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluate
import config


# ─── TC-EVL-01: Module Imports Successfully ───
def test_evaluate_module_imports():
    """evaluate.py must import without errors."""
    assert evaluate is not None


# ─── TC-EVL-02: report_to_per_class_accuracy Returns Dict ───
def test_report_to_per_class_accuracy():
    """Must extract recall values per class from sklearn report dict."""
    report = {
        'Apple___healthy': {'recall': 0.95, 'precision': 0.92, 'f1-score': 0.93},
        'Tomato___healthy': {'recall': 0.88, 'precision': 0.90, 'f1-score': 0.89},
        'accuracy': 0.91,
        'macro avg': {'recall': 0.915},
        'weighted avg': {'recall': 0.91}
    }
    class_names = ['Apple___healthy', 'Tomato___healthy']
    result = evaluate.report_to_per_class_accuracy(report, class_names)
    
    assert isinstance(result, dict)
    assert len(result) == 2
    assert result['Apple___healthy'] == 0.95
    assert result['Tomato___healthy'] == 0.88


# ─── TC-EVL-03: report_to_per_class_accuracy Handles Missing Class ───
def test_report_to_per_class_accuracy_missing_class():
    """Must return 0.0 for classes not present in report."""
    report = {'Apple___healthy': {'recall': 0.95}}
    class_names = ['Apple___healthy', 'Missing_Class']
    result = evaluate.report_to_per_class_accuracy(report, class_names)
    assert result['Missing_Class'] == 0.0


# ─── TC-EVL-04: plot_confusion_matrix Creates PNG ───
def test_plot_confusion_matrix_creates_png(tmp_path):
    """Must save a confusion matrix heatmap as PNG."""
    cm = np.array([[5, 1], [2, 8]])
    class_names = ['Class_A', 'Class_B']
    path = evaluate.plot_confusion_matrix(cm, class_names, tmp_path)
    
    assert Path(path).exists()
    assert Path(path).suffix == '.png'
    assert Path(path).name == 'confusion_matrix.png'


# ─── TC-EVL-05: plot_per_class_accuracy Creates PNG ───
def test_plot_per_class_accuracy_creates_png(tmp_path):
    """Must save a per-class accuracy bar chart as PNG."""
    per_class = {'Class_A': 0.95, 'Class_B': 0.88}
    class_names = ['Class_A', 'Class_B']
    path = evaluate.plot_per_class_accuracy(per_class, class_names, tmp_path)
    
    assert Path(path).exists()
    assert Path(path).suffix == '.png'
    assert Path(path).name == 'per_class_accuracy.png'


# ─── TC-EVL-06: evaluate_model Raises if Model Missing ───
def test_evaluate_model_raises_if_model_missing():
    """Must raise FileNotFoundError if model file does not exist."""
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate_model(model_path="nonexistent_model_12345.h5")


# ─── TC-EVL-07: evaluate_model Function Exists ───
def test_evaluate_function_exists():
    """evaluate_model must be callable."""
    assert callable(evaluate.evaluate_model)


# ─── TC-EVL-08: plot_confusion_matrix Handles Large Matrix ───
def test_plot_confusion_matrix_large(tmp_path):
    """Must handle 38x38 confusion matrix without error."""
    cm = np.random.randint(0, 100, (38, 38))
    class_names = [f'Class_{i}' for i in range(38)]
    path = evaluate.plot_confusion_matrix(cm, class_names, tmp_path, figsize=(20, 18))
    assert Path(path).exists()


# ─── TC-EVL-09: plot_per_class_accuracy Handles Many Classes ───
def test_plot_per_class_accuracy_many_classes(tmp_path):
    """Must handle 38 classes without error."""
    per_class = {f'Class_{i}': np.random.random() for i in range(38)}
    class_names = list(per_class.keys())
    path = evaluate.plot_per_class_accuracy(per_class, class_names, tmp_path)
    assert Path(path).exists()


# ─── TC-EVL-10: Per-Class Accuracy Values in Range ───
def test_per_class_accuracy_values_in_range():
    """Extracted per-class accuracies must be in [0, 1]."""
    report = {
        f'Class_{i}': {'recall': np.random.random()} for i in range(10)
    }
    class_names = [f'Class_{i}' for i in range(10)]
    result = evaluate.report_to_per_class_accuracy(report, class_names)
    for val in result.values():
        assert 0.0 <= val <= 1.0