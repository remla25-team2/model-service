from flask import Flask, jsonify, request
from lib_ml.preprocessing import TextPreprocessor
from lib_version.version_util import VersionUtil
import joblib
import logging
import os 

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to artifacts included in the image
MODEL_PATH = "../models/SentimentModel.pkl"
VECTORIZER_PATH = "../bow/c1_BoW_Sentiment_Model.pkl"

def load_model_and_preprocessor():
    """Load the model and preprocessor from included artifacts."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = TextPreprocessor.load(VECTORIZER_PATH)
        logger.info("Successfully loaded model and preprocessor")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Failed to load model or preprocessor: {e}")
        raise

# Load model and preprocessor at startup
model, preprocessor = load_model_and_preprocessor()

@app.route("/predict", methods=["GET"])
def predict():
    text = request.args.get("text", "")
    X_sparse = preprocessor.transform([text])
    X = X_sparse.toarray()  
    score = model.predict(X)[0]
    return jsonify(sentiment=int(score))


@app.route("/check_health", methods=["GET"])
def check_health():
	return 'OK', 200

@app.route("/version", methods=["GET"])
def version():
    """
    Returns the version of this service as reported by lib-version.
    """
    return jsonify(service_version=VersionUtil.get_version())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
