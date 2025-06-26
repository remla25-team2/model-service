"""
Flask application for sentiment analysis using a pre-trained model.
"""
import logging
import os
from pathlib import Path

import joblib
import requests
from flask import Flask, jsonify, request
from lib_ml.preprocessing import TextPreprocessor
from lib_version.version_util import VersionUtil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to artifacts
MODEL_PATH = "../models/SentimentModel.pkl"
VECTORIZER_PATH = "../bow/BoW_Sentiment_Model.pkl"

# Environment variables for model version
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest")
MODEL_REPO = os.environ.get("MODEL_REPO", "remla25-team2/model-training")

def download_file(url, filepath):
    """Download a file from URL to filepath."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded %s", filepath)
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        raise

def download_model_artifacts():
    """Download model artifacts from GitHub release if they don't exist."""
    model_exists = os.path.exists(MODEL_PATH)
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)

    if model_exists and vectorizer_exists:
        logger.info("Model artifacts already exist, skipping download")
        return

    if MODEL_VERSION == "latest":
        base_url = f"https://github.com/{MODEL_REPO}/releases/latest/download"
    else:
        base_url = f"https://github.com/{MODEL_REPO}/releases/download/v{MODEL_VERSION}"

    if not model_exists:
        model_url = f"{base_url}/SentimentModel.pkl"
        download_file(model_url, MODEL_PATH)

    if not vectorizer_exists:
        vectorizer_url = f"{base_url}/BoW_Sentiment_Model.pkl"
        download_file(vectorizer_url, VECTORIZER_PATH)


def load_model_and_preprocessor():
    """Load the model and preprocessor from artifacts."""
    try:
        # Download artifacts if needed
        download_model_artifacts()

        sentiment_model = joblib.load(MODEL_PATH)
        text_preprocessor = TextPreprocessor.load(VECTORIZER_PATH)
        logger.info("Successfully loaded model and preprocessor")
        return sentiment_model, text_preprocessor
    except Exception as e:
        logger.error("Failed to load model or preprocessor: %s", e)
        raise

# Load model and preprocessor at startup
model, preprocessor = load_model_and_preprocessor()

@app.route("/predict", methods=["GET"])
def predict():
    """Predict sentiment for given text."""
    text = request.args.get("text", "")
    
    try:
        processed_text = preprocessor.process_item(text)
        x_sparse = preprocessor.transform([processed_text])
        x_array = x_sparse.toarray()
        
        # Log feature extraction info for debugging
        feature_count = x_array.shape[1] if len(x_array.shape) > 1 else 0
        non_zero_features = int(sum(x_array[0] != 0)) if len(x_array) > 0 and len(x_array[0]) > 0 else 0
        logger.info(f"Text: '{text[:50]}...', Features: {non_zero_features}/{feature_count}")

        # Get both prediction and probabilities
        prediction = model.predict(x_array)[0]
        probabilities = model.predict_proba(x_array)[0]
        
        # Safety check for empty probabilities
        if len(probabilities) == 0:
            logger.error("Model returned empty probabilities array")
            return jsonify(
                error="Model prediction failed",
                sentiment=0,
                confidence=0.0
            ), 500

        # Calculate confidence as the maximum probability
        confidence = float(max(probabilities))
        
        # Log suspicious predictions for debugging
        if confidence > 0.99:
            logger.warning(f"High confidence prediction: text='{text}', prediction={prediction}, confidence={confidence:.6f}, features_active={non_zero_features}")
        
        # Detect potential vocabulary mismatch issues
        if non_zero_features == 0 and len(text.strip()) > 0:
            logger.warning(f"No features extracted for non-empty text: '{text}' - possible vocabulary mismatch")
            # Still return the prediction but flag it
            return jsonify(
                sentiment=int(prediction),
                confidence=confidence,
                warning="Low feature extraction - possible vocabulary mismatch"
            )
        
        return jsonify(
            sentiment=int(prediction),
            confidence=confidence,
            debug_info={
                "features_extracted": non_zero_features,
                "total_features": feature_count
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error for text '{text}': {e}")
        return jsonify(
            error=f"Prediction failed: {str(e)}",
            sentiment=0,
            confidence=0.0
        ), 500


@app.route("/check_health", methods=["GET"])
def check_health():
    """Health check endpoint."""
    return 'OK', 200

@app.route("/version", methods=["GET"])
def version():
    """
    Returns the version of this service as reported by lib-version.
    """
    return jsonify(service_version=VersionUtil.get_version())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
    