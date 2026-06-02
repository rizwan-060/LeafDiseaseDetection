"""
Data loading module for Leaf Disease Detection.
Connects the dataset directory structure to the augmentation pipeline.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.preprocessing.image import DirectoryIterator

import config
import augmentation


def load_train_generator() -> DirectoryIterator:
    """
    Loads training data with rotation/zoom augmentation applied.
    
    Returns:
        DirectoryIterator: Yields batches of augmented images and labels.
        
    Raises:
        FileNotFoundError: If config.TRAIN_DIR does not exist.
    """
    if not config.TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {config.TRAIN_DIR}")
    
    train_gen = augmentation.get_training_augmentation()
    return train_gen.flow_from_directory(
        directory=str(config.TRAIN_DIR),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED,
        shuffle=True
    )


def load_validation_generator() -> DirectoryIterator:
    """
    Loads validation data WITHOUT augmentation.
    Shuffle is disabled to ensure consistent metrics across epochs.
    
    Returns:
        DirectoryIterator: Yields batches of raw images and labels.
        
    Raises:
        FileNotFoundError: If config.VALID_DIR does not exist.
    """
    if not config.VALID_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {config.VALID_DIR}")
    
    val_gen = augmentation.get_validation_augmentation()
    return val_gen.flow_from_directory(
        directory=str(config.VALID_DIR),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED,
        shuffle=False
    )


def load_test_generator() -> DirectoryIterator:
    """
    Loads test data WITHOUT augmentation.
    Falls back to validation data if test directory is missing.
    """
    target_dir = config.TEST_DIR if config.TEST_DIR.exists() else config.VALID_DIR
    test_gen = augmentation.get_test_augmentation()
    return test_gen.flow_from_directory(
        directory=str(target_dir),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED,
        shuffle=False
    )
    """
    Loads test data WITHOUT augmentation.
    
    Returns:
        DirectoryIterator: Yields batches of raw images and labels.
        
    Raises:
        FileNotFoundError: If config.TEST_DIR does not exist.
    """
    if not config.TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {config.TEST_DIR}")
    
    test_gen = augmentation.get_test_augmentation()
    return test_gen.flow_from_directory(
        directory=str(config.TEST_DIR),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        seed=config.SEED,
        shuffle=False
    )


def get_num_classes() -> int:
    """
    Returns the number of classes found in the training directory.
    
    Returns:
        int: Number of disease/healthy categories.
    """
    return len(config.get_class_names_from_directory(config.TRAIN_DIR))


def get_class_indices() -> Dict[str, int]:
    """
    Returns the class-to-index mapping that Keras uses.
    Keras sorts class names alphabetically.
    
    Returns:
        Dict[str, int]: Mapping of class name -> integer index.
    """
    class_names = config.get_class_names_from_directory(config.TRAIN_DIR)
    return {name: idx for idx, name in enumerate(class_names)}


def count_images(directory: Path) -> int:
    """
    Counts total images in a directory tree recursively.
    
    Args:
        directory: Root path to search.
        
    Returns:
        int: Total number of image files found.
    """
    if not directory.exists():
        return 0
    
    count = 0
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        count += len(list(directory.rglob(pattern)))
    return count


def get_dataset_stats() -> Dict[str, int]:
    """
    Returns image counts for all three splits.
    
    Returns:
        Dict with keys 'train', 'valid', 'test'.
    """
    return {
        "train": count_images(config.TRAIN_DIR),
        "valid": count_images(config.VALID_DIR),
        "test": count_images(config.TEST_DIR),
    }