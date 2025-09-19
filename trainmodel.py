import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import joblib

def train_fraud_model(full_dataset):
    y = full_dataset['is_fraud']
    X = full_dataset.drop(['is_fraud'], axis=1)
    
    columns_to_drop = ['transaction_id', 'user_id', 'timestamp']
    X = X.drop(columns_to_drop, axis=1)
    
    categorical_features = ['card_type', 'location', 'device_fingerprint']
    numerical_features = X.columns.difference(categorical_features)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(X[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    X = pd.concat([X[numerical_features].reset_index(drop=True), encoded_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
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
    joblib.dump(model, filename)
    print(f"\nModel saved successfully to '{filename}'.")

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
