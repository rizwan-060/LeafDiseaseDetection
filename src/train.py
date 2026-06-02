"""
Training Script for Leaf Disease Detection.
Wires data_loader -> augmentation -> model and executes the full training loop.
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

import config
import data_loader
import model as model_module


def get_callbacks() -> list:
    """
    Returns a list of Keras callbacks for training.
    
    Returns:
        list: [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
    """
    # Save only the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        filepath=str(config.MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    # Stop training if validation accuracy stops improving
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when validation loss plateaus
    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    return [checkpoint, early_stop, lr_reduce]


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    fine_tune: bool = False,
    fine_tune_epochs: int = 10
) -> Dict:
    """
    Executes the full training pipeline.
    
    Pipeline:
        1. Load train/validation generators with augmentation
        2. Build model with frozen base (transfer learning)
        3. Train initial phase
        4. Optional: Fine-tune by unfreezing base layers
        5. Save final model and training plots
    
    Args:
        epochs: Number of epochs for initial training.
        batch_size: Batch size for generators.
        learning_rate: Initial learning rate.
        fine_tune: Whether to run fine-tuning phase after initial training.
        fine_tune_epochs: Epochs for fine-tuning.
        
    Returns:
        Dict: Training history and final metrics.
    """
    print("=" * 60)
    print("LEAF DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # ─── Load Data ───
    print("\n[1/4] Loading dataset...")
    train_gen = data_loader.load_train_generator()
    val_gen = data_loader.load_validation_generator()
    
    num_classes = train_gen.num_classes
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    
        # ─── Build Model ───
    print("\n[2/4] Building model (frozen base)...")
    cnn_model = model_module.build_model(
        num_classes=num_classes,
        learning_rate=learning_rate,
        trainable_base=False
    )
    print(f"  Total params: {cnn_model.count_params():,}")
    print(f"  Trainable params: {sum([np.prod(w.shape) for w in cnn_model.trainable_weights]):,}")
    
        
    # ─── Initial Training ───
    print(f"\n[3/4] Training (max {epochs} epochs)...")
    callbacks = get_callbacks()
    
    history = cnn_model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # ─── Optional Fine-Tuning ───
    if fine_tune:
        print(f"\n[3.5/4] Fine-tuning (unfreezing base, {fine_tune_epochs} epochs)...")
        cnn_model = model_module.unfreeze_base_layers(cnn_model, num_layers=30)
        
        history_fine = cnn_model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        # Combine histories
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    # ─── Save & Plot ───
    print("\n[4/4] Saving artifacts...")
    final_path = model_module.save_model(cnn_model)
    print(f"  Model saved to: {final_path}")
    
    plot_path = plot_training_history(history)
    print(f"  History plot saved to: {plot_path}")
    
    # ─── Final Metrics ───
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    print(f"  Final Val Accuracy:   {final_val_acc:.4f}")
    print(f"  Final Train Loss:     {final_train_loss:.4f}")
    print(f"  Final Val Loss:       {final_val_loss:.4f}")
    print("=" * 60)
    
    return {
        'history': history.history,
        'model_path': final_path,
        'plot_path': plot_path,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss
    }


def plot_training_history(history, save_dir: Path = config.MODEL_DIR) -> str:
    """
    Plots training & validation accuracy and loss curves.
    
    Args:
        history: Keras History object or dict with 'history' attribute.
        save_dir: Directory to save the plot.
        
    Returns:
        str: Path to saved plot image.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    hist = history.history if hasattr(history, 'history') else history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(hist['accuracy'], label='Train Accuracy', color='blue')
    axes[0].plot(hist['val_accuracy'], label='Val Accuracy', color='orange')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(hist['loss'], label='Train Loss', color='blue')
    axes[1].plot(hist['val_loss'], label='Val Loss', color='orange')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return str(plot_path)


if __name__ == "__main__":
    # Run full training when executed directly
    results = train()