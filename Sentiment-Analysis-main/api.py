from flask import Flask, request, jsonify
import pickle
import logging
from flask_cors import CORS

# Initialize app and logging
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the trained model pipeline
try:
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    app.logger.info("✅ Model loaded successfully.")
except Exception as e:
    app.logger.error("❌ Error loading model: %s", str(e))
    model = None

# Label mapping for predictions
label_map = {0: "negative", 1: "positive", 2: "neutral"}

@app.route('/')
def home():
    return "✅ Sentiment Analysis API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()

        # Validate input
        if not data or 'text' not in data or not data['text'].strip():
            return jsonify({'error': '❗ Please provide a non-empty "text" field.'}), 400

        text = data['text']

        # Make prediction
        prediction = model.predict([text])[0]
        prediction_proba = model.predict_proba([text])[0]

        result = {
            'text': text,
            'prediction': label_map.get(prediction, "unknown"),
            'probabilities': {
                'negative': round(float(prediction_proba[0]), 3),
                'positive': round(float(prediction_proba[1]), 3),
                'neutral': round(float(prediction_proba[2]), 3)
            }
        }

        app.logger.info("✅ Prediction done for input: %s", text)
        return jsonify(result)

    except Exception as e:
        app.logger.error("❌ Error during prediction: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
