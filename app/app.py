import os
from flask import Flask, request, jsonify, render_template
from lib_ml import preprocess

app = Flask(__name__)

PORT = int(os.environ.get('PORT', 5001))
MODEL_SERVICE_VERSION = os.environ.get('MODEL_SERVICE_VERSION', '0.1.0')

@app.route('/version/modelversion')
def version():
    """
    Returns the version of the model-service.
    """
    return jsonify({'model_service_version': MODEL_SERVICE_VERSION})

@app.route('/predict', methods=['POST'])
def predict():
    """
    A dummy /predict endpoint:
    """
    data = request.get_json(force=True)
    text = data.get('text', '')
    features = preprocess(text)            # e.g. [len(text)]
    label = 'positive' if features[0] > 5 else 'negative'
    return jsonify({'prediction': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
