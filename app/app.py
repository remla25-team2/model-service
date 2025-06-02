"""
Flask application for sentiment analysis using a pre-trained model.
"""
import logging
import os

import joblib
from flask import Flask, jsonify, request
from lib_ml.preprocessing import TextPreprocessor
from lib_version.version_util import VersionUtil

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to artifacts
MODEL_PATH = "../models/SentimentModel.pkl"
VECTORIZER_PATH = "../bow/c1_BoW_Sentiment_Model.pkl"

# Environment variables for model version
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest")
MODEL_REPO = os.environ.get("MODEL_REPO", "remla25-team2/model-training")

def download_file(url, filepath):
    """Download a file from URL to filepath."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded {filepath}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
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
        vectorizer_url = f"{base_url}/c1_BoW_Sentiment_Model.pkl"
        download_file(vectorizer_url, VECTORIZER_PATH)


def load_model_and_preprocessor():
    """Load the model and preprocessor from included artifacts."""
    try:
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
    x_sparse = preprocessor.transform([text])
    x_array = x_sparse.toarray()
    score = model.predict(x_array)[0]
    return jsonify(sentiment=int(score))


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
    