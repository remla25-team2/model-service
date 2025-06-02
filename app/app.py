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

# Paths to artifacts included in the image
MODEL_PATH = "../models/SentimentModel.pkl"
VECTORIZER_PATH = "../bow/c1_BoW_Sentiment_Model.pkl"

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
    