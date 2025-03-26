from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load all models
try:
    logger.info("Loading models...")
    slr_model = joblib.load('models/slr_model.joblib')
    mlr_model = joblib.load('models/mlr_model.joblib')
    poly_model, poly_reg = joblib.load('models/poly_model.joblib')
    logistic_model = joblib.load('models/logistic_model.joblib')
    knn_model = joblib.load('models/knn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    logger.info("All models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        age = float(data['age'])
        experience = float(data['experience'])
        education_level = float(data['education_level'])
        model_type = data['model_type']

        # Validate input
        if not (0 <= age <= 100 and 0 <= experience <= 50 and 1 <= education_level <= 4):
            return jsonify({'error': 'Invalid input values'}), 400

        # Prepare input data
        input_data = np.array([[age, experience, education_level]])
        input_scaled = scaler.transform(input_data)

        if model_type == 'slr':
            # Simple Linear Regression (using only experience)
            prediction = slr_model.predict([[experience]])[0]
        elif model_type == 'mlr':
            # Multiple Linear Regression
            prediction = mlr_model.predict(input_scaled)[0]
        elif model_type == 'poly':
            # Polynomial Regression
            input_poly = poly_model.transform(input_scaled)
            prediction = poly_reg.predict(input_poly)[0]
        elif model_type == 'logistic':
            # Logistic Regression
            prediction = logistic_model.predict(input_scaled)[0]
        elif model_type == 'knn':
            # KNN
            prediction = knn_model.predict(input_scaled)[0]
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        logger.info(f"Prediction successful: {prediction}")
        return jsonify({
            'prediction': float(prediction),
            'model_type': model_type
        })

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 