from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = ['card_type', 'location', 'device_fingerprint']
LOCATIONS = ['New York, NY', 'San Francisco, CA', 'Los Angeles, CA', 'Chicago, IL', 'Miami, FL', 'Houston, TX']
CARD_TYPES = ['Visa', 'Mastercard', 'Amex']
DEVICE_MODELS = ['iPhone 14', 'Galaxy S23', 'Pixel 7', 'MacBook Air', 'iPad Pro']

app = Flask(__name__)
CORS(app)

def load_model_and_encoder():
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

@app.route('/predict', methods=['POST'])
def predict_transaction():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    transaction_data = request.get_json()
    print(f"Received transaction data for prediction: {transaction_data}")

    if model is None or encoder is None:
        return jsonify({"error": "Model or encoder not loaded. Cannot make prediction."}), 500

    try:
        new_df = pd.DataFrame([transaction_data])
        
        encoded_features = encoder.transform(new_df[CATEGORICAL_FEATURES])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        
        numerical_features = new_df.columns.difference(CATEGORICAL_FEATURES)
        final_df = pd.concat([new_df[numerical_features].reset_index(drop=True), encoded_df], axis=1)

        prediction = model.predict(final_df)
        
        if prediction[0] == 1:
            return jsonify({"prediction": "fraudulent"})
        else:
            return jsonify({"prediction": "legitimate"})
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == '__main__':
    app.run(debug=True)
