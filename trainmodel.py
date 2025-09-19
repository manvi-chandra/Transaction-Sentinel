import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import joblib

def train_fraud_model(full_dataset):
    """
    Trains a fraud detection model using the generated dataset.

    This function preprocesses the data by handling categorical features and then
    trains a RandomForestClassifier.

    Args:
        full_dataset (pandas.DataFrame): The combined transaction dataset.
    Returns:
        Trained model (sklearn.ensemble.RandomForestClassifier): The trained model object.
    """
    print("\n--- Starting Data Preprocessing ---")
    
    # 1. Separate features (X) and target (y)
    y = full_dataset['is_fraud']
    X = full_dataset.drop(['is_fraud'], axis=1)
    
    # Drop columns that are not useful for training
    columns_to_drop = ['transaction_id', 'user_id', 'timestamp']
    X = X.drop(columns_to_drop, axis=1)
    
    # 2. Identify categorical and numerical columns
    categorical_features = ['card_type', 'location', 'device_fingerprint']
    numerical_features = X.columns.difference(categorical_features)
    
    # 3. Handle categorical features with One-Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # 4. Combine encoded features with numerical features
    X = pd.concat([X[numerical_features].reset_index(drop=True), encoded_df], axis=1)

    print("\n--- Training Model ---")
    # 5. Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Train the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 7. Evaluate the model's performance
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("  - True Positives (TP): Correctly identified fraudulent transactions.")
    print("  - True Negatives (TN): Correctly identified legitimate transactions.")
    print("  - False Positives (FP): Legitimate transactions incorrectly flagged as fraud.")
    print("  - False Negatives (FN): Fraudulent transactions incorrectly identified as legitimate.")
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(class_report)
    
    print(f"ROC AUC Score: {roc_auc:.4f}")

    print("\nModel training and evaluation are complete. The trained model can now be used for making predictions on new data.")
    
    return model

def save_model(model, filename="fraud_model.joblib"):
    """
    Saves a trained model to a file.

    Args:
        model: The trained model object.
        filename (str): The name of the file to save the model to.
    """
    joblib.dump(model, filename)
    print(f"\nModel saved successfully to '{filename}'.")

# Main block to load the data, train the model, and save it
if __name__ == "__main__":
    file_path = 'transaction_data.csv'
    
    if os.path.exists(file_path):
        print(f"Loading data from '{file_path}'...")
        try:
            full_dataset = pd.read_csv(file_path)
            trained_model = train_fraud_model(full_dataset)
            save_model(trained_model)
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please run 'generate_dataset.py' first to create the dataset.")
