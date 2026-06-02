"""
Inference Engine for Leaf Disease Detection.
Loads a trained model and predicts the disease class for a single image.
Includes treatment recommendation mapping.
"""

from pathlib import Path
from typing import Union, Dict, List

import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

import config
import preprocessor


# ─── Treatment Recommendation Database ───
TREATMENT_MAP: Dict[str, str] = {
    'Apple___Apple_scab': 'Apply fungicide (Mancozeb). Remove and destroy infected leaves.',
    'Apple___Black_rot': 'Prune cankers. Apply copper fungicide. Remove mummified fruits.',
    'Apple___Cedar_apple_rust': 'Remove cedar galls nearby. Apply fungicide early spring.',
    'Apple___healthy': 'No treatment needed. Maintain regular care and monitoring.',
    'Blueberry___healthy': 'No treatment needed. Maintain soil pH 4.5–5.5.',
    'Cherry___Powdery_mildew': 'Apply sulfur-based fungicide. Improve air circulation.',
    'Cherry___healthy': 'No treatment needed. Prune for airflow.',
    'Corn___Cercospora_leaf_spot': 'Apply fungicide (Azoxystrobin). Rotate crops annually.',
    'Corn___Common_rust': 'Plant resistant varieties. Apply fungicide if severe.',
    'Corn___Northern_Leaf_Blight': 'Rotate crops. Remove debris. Apply fungicide if needed.',
    'Corn___healthy': 'No treatment needed. Maintain balanced fertilization.',
    'Grape___Black_rot': 'Remove mummified berries. Apply fungicide (Mancozeb).',
    'Grape___Esca_(Black_Measles)': 'Prune infected wood. No chemical cure available.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply Bordeaux mixture. Remove infected leaves.',
    'Grape___healthy': 'No treatment needed. Maintain canopy management.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure. Remove infected tree to prevent spread.',
    'Peach___Bacterial_spot': 'Apply copper spray. Avoid overhead irrigation.',
    'Peach___healthy': 'No treatment needed. Prune during dormancy.',
    'Pepper,_bell___Bacterial_spot': 'Use disease-free seeds. Apply copper fungicide.',
    'Pepper,_bell___healthy': 'No treatment needed. Ensure proper spacing.',
    'Potato___Early_blight': 'Apply fungicide (Chlorothalonil). Rotate crops 3+ years.',
    'Potato___Late_blight': 'Apply fungicide immediately. Destroy infected plants.',
    'Potato___healthy': 'No treatment needed. Hill soil around stems.',
    'Raspberry___healthy': 'No treatment needed. Prune old canes annually.',
    'Soybean___healthy': 'No treatment needed. Rotate with corn or wheat.',
    'Squash___Powdery_mildew': 'Apply neem oil or sulfur. Plant resistant varieties.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Apply fungicide if severe.',
    'Strawberry___healthy': 'No treatment needed. Mulch to prevent fruit rot.',
    'Tomato___Bacterial_spot': 'Apply copper spray. Avoid working with wet plants.',
    'Tomato___Early_blight': 'Remove lower infected leaves. Apply fungicide.',
    'Tomato___Late_blight': 'Destroy infected plants. Apply fungicide (Mancozeb).',
    'Tomato___Leaf_Mold': 'Reduce humidity. Apply fungicide. Improve ventilation.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Apply fungicide weekly.',
    'Tomato___Spider_mites': 'Apply miticide or insecticidal soap. Increase humidity.',
    'Tomato___Target_Spot': 'Remove infected debris. Apply fungicide (Chlorothalonil).',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure. Remove infected plant. Control whiteflies.',
    'Tomato___Tomato_mosaic_virus': 'No cure. Remove infected plant. Disinfect tools.',
    'Tomato___healthy': 'No treatment needed. Stake plants for airflow.',
    'Background_without_leaves': 'No leaf detected. Please upload a clear leaf image.'
}


def get_treatment_recommendation(disease_name: str) -> str:
    """
    Returns a treatment recommendation for the predicted disease class.
    
    Args:
        disease_name: The predicted class name.
        
    Returns:
        str: Treatment advice string.
    """
    return TREATMENT_MAP.get(
        disease_name,
        'Consult your local agricultural extension office for treatment advice.'
    )


def load_model(path: str = None):
    """
    Loads a saved Keras model from disk.
    
    Args:
        path: Path to .h5 model file. Defaults to config.MODEL_PATH.
        
    Returns:
        tf.keras.Model: Loaded model ready for inference.
        
    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if path is None:
        path = str(config.MODEL_PATH)
    
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train the model first using src/train.py"
        )
    
    return keras_load_model(path)


def predict_image(
    image_path: Union[str, Path],
    model=None,
    top_k: int = None
) -> Dict:
    """
    Full prediction pipeline for a single image file.
    
    Pipeline:
        1. Load model (if not provided)
        2. Preprocess image
        3. Predict probabilities
        4. Extract top-k classes
        5. Map to treatment recommendations
    
    Args:
        image_path: Path to the image file.
        model: Pre-loaded Keras model (optional).
        top_k: Number of top predictions to return. Defaults to config.TOP_K_PREDICTIONS.
        
    Returns:
        Dict: {
            'predicted_class': str,
            'confidence': float,
            'is_healthy': bool,
            'uncertain': bool,
            'top_k': List[Dict],
            'all_probabilities': Dict[str, float]
        }
    """
    if top_k is None:
        top_k = config.TOP_K_PREDICTIONS
    
    if model is None:
        model = load_model()
    
    # Preprocess
    img = preprocessor.preprocess_image(str(image_path))
    img_batch = preprocessor.add_batch_dimension(img)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    probs = predictions[0]
    
    # Ensure class names are loaded
    if not config.CLASS_NAMES:
        config.refresh_class_names()
    
    class_names = config.CLASS_NAMES if config.CLASS_NAMES else [f"Class_{i}" for i in range(len(probs))]
    
    # Get top-k indices (highest probability first)
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    top_k_results = []
    for idx in top_indices:
        class_name = class_names[idx]
        top_k_results.append({
            'class': class_name,
            'confidence': float(probs[idx]),
            'treatment': get_treatment_recommendation(class_name)
        })
    
    predicted_class = top_k_results[0]['class']
    confidence = top_k_results[0]['confidence']
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_healthy': 'healthy' in predicted_class.lower(),
        'uncertain': confidence < config.CONFIDENCE_THRESHOLD,
        'top_k': top_k_results,
        'all_probabilities': {class_names[i]: float(probs[i]) for i in range(len(probs))}
    }


def predict_array(
    image_array: np.ndarray,
    model=None,
    top_k: int = None
) -> Dict:
    """
    Prediction pipeline for an in-memory numpy array.
    Same as predict_image but accepts a pre-loaded array instead of a file path.
    
    Args:
        image_array: RGB image as numpy array (H, W, 3).
        model: Pre-loaded Keras model (optional).
        top_k: Number of top predictions to return.
        
    Returns:
        Dict: Same structure as predict_image.
    """
    if top_k is None:
        top_k = config.TOP_K_PREDICTIONS
    
    if model is None:
        model = load_model()
    
    # Preprocess array
    img = preprocessor.preprocess_array(image_array)
    img_batch = preprocessor.add_batch_dimension(img)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)
    probs = predictions[0]
    
    if not config.CLASS_NAMES:
        config.refresh_class_names()
    
    class_names = config.CLASS_NAMES if config.CLASS_NAMES else [f"Class_{i}" for i in range(len(probs))]
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    top_k_results = []
    for idx in top_indices:
        class_name = class_names[idx]
        top_k_results.append({
            'class': class_name,
            'confidence': float(probs[idx]),
            'treatment': get_treatment_recommendation(class_name)
        })
    
    predicted_class = top_k_results[0]['class']
    confidence = top_k_results[0]['confidence']
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_healthy': 'healthy' in predicted_class.lower(),
        'uncertain': confidence < config.CONFIDENCE_THRESHOLD,
        'top_k': top_k_results,
        'all_probabilities': {class_names[i]: float(probs[i]) for i in range(len(probs))}
    }