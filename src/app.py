"""
Flask Web Application for Leaf Disease Detection.
Provides endpoints for image upload, disease prediction, and health checks.
"""

import sys
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import predictor
import utils


app = Flask(
    __name__,
    template_folder=str(config.BASE_DIR / "templates"),
    static_folder=str(config.BASE_DIR / "static")
)
app.secret_key = "leaf_disease_detection_secret_key_2024"


# ─── Routes ───

@app.route("/")
def index():
    """
    Home page with image upload form.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image upload and returns disease prediction.
    """
    # Check if file part exists
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    
    # Check if user selected a file
    if file.filename == "":
        flash("No file selected. Please choose an image.", "error")
        return redirect(url_for("index"))
    
    # Validate file type
    if not utils.validate_image_file(file.filename):
        flash("Invalid file type. Please upload JPG, JPEG, or PNG.", "error")
        return redirect(url_for("index"))
    
    try:
        # Save uploaded file
        saved_path = utils.save_uploaded_file(file)
        
        # Run prediction
        result = predictor.predict_image(str(saved_path))
        
        # Cleanup old uploads
        utils.cleanup_uploads()
        
        # Encode image for display
        img_data = utils.encode_image_base64(saved_path)
        
        return render_template(
            "result.html",
            image_data=img_data,
            filename=secure_filename(file.filename),
            predicted_class=result["predicted_class"],
            confidence=round(result["confidence"] * 100, 2),
            is_healthy=result["is_healthy"],
            uncertain=result["uncertain"],
            top_k=result["top_k"],
            treatment=result["top_k"][0]["treatment"] if result["top_k"] else "No recommendation available."
        )
        
    except FileNotFoundError as e:
        flash(f"Model not found: {str(e)}. Please train the model first.", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/health")
def health():
    """
    Health check endpoint for monitoring.
    Returns JSON status.
    """
    status = utils.get_health_status()
    code = 200 if status["status"] == "ok" else 503
    return jsonify(status), code


@app.route("/classes")
def list_classes():
    """
    Returns list of supported disease classes as JSON.
    """
    if not config.CLASS_NAMES:
        config.refresh_class_names()
    return jsonify({
        "count": len(config.CLASS_NAMES),
        "classes": config.CLASS_NAMES
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    flash("An internal server error occurred.", "error")
    return redirect(url_for("index")), 500


# ─── Entry Point ───

if __name__ == "__main__":
    print("=" * 60)
    print("LEAF DISEASE DETECTION WEB APP")
    print("=" * 60)
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Dataset: {config.DATA_DIR}")
    print(f"Classes: {len(config.CLASS_NAMES)}")
    print("=" * 60)
    print("Open browser: http://127.0.0.1:5000")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)