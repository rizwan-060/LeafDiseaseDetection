"""
CNN Model Architecture for Leaf Disease Detection.
Uses MobileNetV2 as a transfer learning base with a custom classification head.
Implemented in TensorFlow/Keras as required by the project guidelines.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import config


def build_model(
    num_classes: int = None,
    input_shape: tuple = config.IMG_SHAPE,
    dropout_rate: float = 0.3,
    learning_rate: float = config.LEARNING_RATE,
    trainable_base: bool = False
) -> Model:
    """
    Builds a CNN for leaf disease classification using MobileNetV2 transfer learning.
    
    Architecture:
        1. MobileNetV2 base (pre-trained on ImageNet, frozen by default)
        2. GlobalAveragePooling2D
        3. BatchNormalization
        4. Dropout (regularization)
        5. Dense(256, ReLU)
        6. Dropout
        7. Dense(num_classes, softmax)
    
    Args:
        num_classes: Number of output classes. Defaults to config value.
        input_shape: Input tensor shape (H, W, C).
        dropout_rate: Dropout probability for regularization.
        learning_rate: Adam optimizer learning rate.
        trainable_base: If True, base MobileNetV2 layers are trainable (fine-tuning).
        
    Returns:
        tf.keras.Model: Compiled and ready-to-train model.
    """
    if num_classes is None:
        num_classes = config.get_num_classes() or config.NUM_CLASSES_FALLBACK
    
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    
    # ─── Transfer Learning Base ───
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base
    
    # ─── Custom Classification Head ───
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # training=False for inference mode in base
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate / 2)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name='LeafDisease_MobileNetV2')
    
    # ─── Compile ───
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def unfreeze_base_layers(model: Model, num_layers: int = 30) -> Model:
    """
    Unfreezes the last N layers of the base model for fine-tuning.
    Call this after initial training with frozen base.
    
    Args:
        model: The compiled model.
        num_layers: Number of layers from the end to unfreeze.
        
    Returns:
        tf.keras.Model: Model with updated trainable flags.
    """
    # Find the base model layer (it's the second layer after Input)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model):  # MobileNetV2 is a nested Model
            base_model = layer
            break
    
    if base_model is None:
        raise ValueError("Could not find nested base model in architecture")
    
    # Freeze all first, then unfreeze last N
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Returns a string summary of the model architecture.
    
    Args:
        model: The Keras model.
        
    Returns:
        str: Model summary text.
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def count_trainable_params(model: Model) -> int:
    """
    Counts the number of trainable parameters in the model.
    
    Args:
        model: The Keras model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return model.count_params()


def save_model(model: Model, path: str = None) -> str:
    """
    Saves the model to disk.
    
    Args:
        model: The trained model.
        path: Save path. Defaults to config.MODEL_PATH.
        
    Returns:
        str: Path where model was saved.
    """
    if path is None:
        path = str(config.MODEL_PATH)
    
    model.save(path)
    return path


def load_model(path: str = None) -> Model:
    """
    Loads a saved model from disk.
    
    Args:
        path: Load path. Defaults to config.MODEL_PATH.
        
    Returns:
        tf.keras.Model: Loaded model ready for inference.
    """
    if path is None:
        path = str(config.MODEL_PATH)
    
    return tf.keras.models.load_model(path)