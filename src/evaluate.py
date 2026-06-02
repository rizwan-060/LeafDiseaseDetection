"""
Evaluation Module for Leaf Disease Detection.
Generates confusion matrix, classification report, and per-class metrics
after model training is complete.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import config
import data_loader
import model as model_module


def evaluate_model(
    model_path: str = None,
    output_dir: Path = config.MODEL_DIR
) -> Dict:
    """
    Evaluates a trained model on the validation dataset.
    
    Generates:
        - Classification report (precision, recall, f1 per class)
        - Confusion matrix heatmap
        - Overall accuracy, precision, recall, f1
    
    Args:
        model_path: Path to saved .h5 model. Defaults to config.MODEL_PATH.
        output_dir: Directory to save evaluation plots.
        
    Returns:
        Dict: Evaluation metrics and file paths.
    """
    if model_path is None:
        model_path = str(config.MODEL_PATH)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LEAF DISEASE DETECTION - MODEL EVALUATION")
    print("=" * 60)
    
    # ─── Load Model ───
    print(f"\n[1/3] Loading model from: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    cnn_model = model_module.load_model(model_path)
    print(f"  Model loaded successfully.")
    
    # ─── Load Validation Data ───
    print("\n[2/3] Loading validation data...")
    val_gen = data_loader.load_validation_generator()
    class_names = list(val_gen.class_indices.keys())
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Classes: {len(class_names)}")
    
    # ─── Predict ───
    print("\n[3/3] Running predictions...")
    val_gen.reset()
    y_pred_probs = cnn_model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels (generator yields shuffled=False, so order matches)
    y_true = val_gen.classes
    
    # ─── Overall Metrics ───
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  Overall Accuracy:  {accuracy:.4f}")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall:    {recall:.4f}")
    print(f"  Weighted F1-Score:  {f1:.4f}")
    
    # ─── Classification Report ───
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
    
    # Save report to text file
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"\n  Classification report saved to: {report_path}")
    
    # ─── Confusion Matrix ───
    cm = confusion_matrix(y_true, y_pred)
    cm_path = plot_confusion_matrix(cm, class_names, output_dir)
    print(f"  Confusion matrix saved to: {cm_path}")
    
    # ─── Per-Class Accuracy Bar Chart ───
    per_class_acc = report_to_per_class_accuracy(report, class_names)
    acc_path = plot_per_class_accuracy(per_class_acc, class_names, output_dir)
    print(f"  Per-class accuracy saved to: {acc_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'report_path': str(report_path),
        'cm_path': str(cm_path),
        'acc_path': str(acc_path)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    figsize: tuple = (16, 14)
) -> str:
    """
    Plots a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array (n_classes, n_classes).
        class_names: List of class name strings.
        output_dir: Directory to save plot.
        figsize: Figure size tuple.
        
    Returns:
        str: Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6)
    
    plt.tight_layout()
    save_path = output_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path)


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    class_names: List[str],
    output_dir: Path,
    figsize: tuple = (14, 10)
) -> str:
    """
    Plots a horizontal bar chart of per-class accuracy.
    
    Args:
        per_class_acc: Dict mapping class name -> accuracy.
        class_names: Ordered list of class names.
        output_dir: Directory to save plot.
        figsize: Figure size tuple.
        
    Returns:
        str: Path to saved plot.
    """
    accuracies = [per_class_acc[name] for name in class_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(class_names))
    
    bars = ax.barh(y_pos, accuracies, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_title('Per-Class Accuracy', fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    save_path = output_dir / 'per_class_accuracy.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(save_path)


def report_to_per_class_accuracy(report: Dict, class_names: List[str]) -> Dict[str, float]:
    """
    Extracts per-class accuracy-like metric (recall per class) from sklearn report.
    In classification_report, 'recall' for a class = TP / (TP + FN), which is
    equivalent to per-class accuracy (how many of this class were correctly identified).
    
    Args:
        report: sklearn classification_report output_dict.
        class_names: List of class names.
        
    Returns:
        Dict[str, float]: Class name -> recall (per-class accuracy).
    """
    per_class = {}
    for name in class_names:
        # sklearn report keys are class names as strings
        if name in report:
            per_class[name] = report[name]['recall']
        else:
            per_class[name] = 0.0
    return per_class


if __name__ == "__main__":
    # Run evaluation when executed directly
    results = evaluate_model()