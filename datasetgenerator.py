import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_transaction_data(num_users=1000, num_transactions_per_user=100):
    """
    Generates a synthetic transaction dataset for fraud detection.

    The dataset includes features for user behavior analysis, such as time of day,
    location, and device data. Fraud is simulated in two patterns:
    1. Unusually high transaction amounts.
    2. Transactions from an unexpected location for a given user.

    Args:
        num_users (int): The number of unique users to simulate.
        num_transactions_per_user (int): The average number of transactions per user.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic transaction data.
    """
    print("Generating synthetic transaction dataset...")
    
    locations = ['New York, NY', 'San Francisco, CA', 'Los Angeles, CA', 'Chicago, IL', 'Miami, FL', 'Houston, TX']
    card_types = ['Visa', 'Mastercard', 'Amex']
    device_models = ['iPhone 14', 'Galaxy S23', 'Pixel 7', 'MacBook Air', 'iPad Pro']
    
    data = []
    
    user_profiles = {
        f"user_{i}": {
            'typical_location': random.choice(locations),
            'typical_device': random.choice(device_models)
        } for i in range(num_users)
    }

    for user_id, profile in user_profiles.items():
        num_transactions = int(np.random.normal(num_transactions_per_user, 10))
        for _ in range(num_transactions):
            
            transaction_id = f"txn_{len(data) + 1}"
            timestamp = datetime.now() - timedelta(days=random.randint(1, 365), seconds=random.randint(0, 86400))
            
            is_fraud = False
            amount = round(np.random.lognormal(mean=2.5, sigma=0.5), 2) * 10
            location = profile['typical_location']
            device = profile['typical_device']
            
            if random.random() < 0.03:
                is_fraud = True
                amount = round(np.random.uniform(1000, 5000), 2)
                location = random.choice(locations)
                device = random.choice(device_models)
            
            if random.random() < 0.02:
                is_fraud = True
                amount = round(np.random.lognormal(mean=2.5, sigma=0.5), 2) * 10
                new_location = random.choice(locations)
                while new_location == profile['typical_location']:
                    new_location = random.choice(locations)
                location = new_location
                
            data.append({
                'transaction_id': transaction_id,
                'user_id': user_id,
                'amount': amount,
                'card_type': random.choice(card_types),
                'location': location,
                'device_fingerprint': device,
                'timestamp': timestamp,
                'time_of_day': timestamp.hour,
                'is_fraud': is_fraud
            })

    df = pd.DataFrame(data)
    df.to_csv("transaction_data.csv", index=False)
    print(f"\nSuccessfully generated {len(df)} transactions and saved to 'transaction_data.csv'.")
    
    return df

if __name__ == '__main__':
    generate_transaction_data()
