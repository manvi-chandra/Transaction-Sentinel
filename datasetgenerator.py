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
    
    # 1. Define lists for categorical data
    locations = ['New York, NY', 'San Francisco, CA', 'Los Angeles, CA', 'Chicago, IL', 'Miami, FL', 'Houston, TX']
    card_types = ['Visa', 'Mastercard', 'Amex']
    device_models = ['iPhone 14', 'Galaxy S23', 'Pixel 7', 'MacBook Air', 'iPad Pro']
    
    # 2. Initialize lists to store generated data
    data = []
    
    # 3. Generate a set of typical locations and devices for each user
    user_profiles = {
        f"user_{i}": {
            'typical_location': random.choice(locations),
            'typical_device': random.choice(device_models)
        } for i in range(num_users)
    }

    # 4. Generate transactions for each user
    for user_id, profile in user_profiles.items():
        # Generate transactions for the user, with some variation in count
        num_transactions = int(np.random.normal(num_transactions_per_user, 10))
        for _ in range(num_transactions):
            
            # 5. Simulate transaction details
            transaction_id = f"txn_{len(data) + 1}"
            timestamp = datetime.now() - timedelta(days=random.randint(1, 365), seconds=random.randint(0, 86400))
            
            # Normal transactions: random amount, typical location, typical device
            is_fraud = False
            amount = round(np.random.lognormal(mean=2.5, sigma=0.5), 2) * 10 # Skewed towards smaller amounts
            location = profile['typical_location']
            device = profile['typical_device']
            
            # 6. Introduce fraud patterns
            # Pattern 1: High-value fraud (3% of transactions)
            if random.random() < 0.03:
                is_fraud = True
                amount = round(np.random.uniform(1000, 5000), 2)
                location = random.choice(locations)
                device = random.choice(device_models)
            
            # Pattern 2: Location-based fraud (2% of transactions)
            if random.random() < 0.02:
                # Ensure the new location is different from the user's typical one
                is_fraud = True
                amount = round(np.random.lognormal(mean=2.5, sigma=0.5), 2) * 10
                new_location = random.choice(locations)
                while new_location == profile['typical_location']:
                    new_location = random.choice(locations)
                location = new_location
                
            # Add transaction to the list
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

    # 7. Create the DataFrame and save it
    df = pd.DataFrame(data)
    df.to_csv("transaction_data.csv", index=False)
    print(f"\nSuccessfully generated {len(df)} transactions and saved to 'transaction_data.csv'.")
    
    return df

if __name__ == '__main__':
    generate_transaction_data()
