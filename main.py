from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

# --- Global Configurations and Data Structures ---
CATEGORICAL_FEATURES = ['card_type', 'location', 'device_fingerprint']
# These lists must match the ones used in the dataset generator script
LOCATIONS = ['New York, NY', 'San Francisco, CA', 'Los Angeles, CA', 'Chicago, IL', 'Miami, FL', 'Houston, TX']
CARD_TYPES = ['Visa', 'Mastercard', 'Amex']
DEVICE_MODELS = ['iPhone 14', 'Galaxy S23', 'Pixel 7', 'MacBook Air', 'iPad Pro']

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS to allow the frontend to communicate with this backend
CORS(app) 

# --- Model and Encoder Loading ---
# Load the trained model and fitted encoder from files.
# This is a one-time operation when the application starts.
def load_model_and_encoder():
    """
    Loads the trained model and One-Hot Encoder from files.
    """
    try:
        model = joblib.load("fraud_model.joblib")
        encoder = joblib.load("encoder.joblib")
        print("Model and encoder loaded successfully.")
        return model, encoder
    except FileNotFoundError:
        print("Error: Model or encoder file not found.")
        print("Please ensure 'fraud_model.joblib' and 'encoder.joblib' are in the same directory.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model or encoder: {e}")
        return None, None

model, encoder = load_model_and_encoder()
if model is None or encoder is None:
    print("Application cannot run without a valid model. Please train one first.")

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict_transaction():
    """
    Receives a new transaction from the frontend and returns a prediction.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    transaction_data = request.get_json()
    print(f"Received transaction data for prediction: {transaction_data}")

    if model is None or encoder is None:
        return jsonify({"error": "Model or encoder not loaded. Cannot make prediction."}), 500

    try:
        # Preprocess the incoming transaction data
        new_df = pd.DataFrame([transaction_data])
        
        # Transform categorical features using the pre-fitted encoder
        encoded_features = encoder.transform(new_df[CATEGORICAL_FEATURES])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        
        # Combine numerical and encoded features
        numerical_features = new_df.columns.difference(CATEGORICAL_FEATURES)
        final_df = pd.concat([new_df[numerical_features].reset_index(drop=True), encoded_df], axis=1)

        # Make a prediction
        prediction = model.predict(final_df)
        
        # Return the prediction as a JSON response
        if prediction[0] == 1:
            return jsonify({"prediction": "fraudulent"})
        else:
            return jsonify({"prediction": "legitimate"})
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

# --- Main block to run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True)
