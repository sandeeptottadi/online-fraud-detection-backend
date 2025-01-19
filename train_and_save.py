import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Type mapping
TYPE_MAPPING = {
    "PAYMENT": 0,
    "CASH_IN": 1,
    "DEBIT": 2,
    "CASH_OUT": 3,
    "TRANSFER": 4
}

def generate_transaction():
    is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])
    
    if is_fraud:
        step = np.random.randint(200, 220)
        type = np.random.choice(['TRANSFER', 'CASH_OUT'], p=[0.7, 0.3])
        amount = np.random.choice([
            np.random.uniform(0.1, 10.0),
            np.random.uniform(50000, 1000000)
        ], p=[0.6, 0.4])
        oldbalanceOrg = np.random.choice([
            0.0,
            amount * 0.1,
            amount * 0.5
        ], p=[0.4, 0.4, 0.2])
        oldbalanceDest = np.random.choice([
            0.0,
            amount * 0.1
        ], p=[0.8, 0.2])
    else:
        step = np.random.randint(1, 500)
        type = np.random.choice(['PAYMENT', 'CASH_IN', 'DEBIT', 'TRANSFER', 'CASH_OUT'])
        amount = np.random.uniform(100, 50000)
        oldbalanceOrg = amount * np.random.uniform(2, 5)
        oldbalanceDest = np.random.uniform(1000, 100000)
    
    return {
        'step': step,
        'type': type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'oldbalanceDest': oldbalanceDest,
        'isFraud': is_fraud
    }

def main():
    # Generate dataset
    n_samples = 10000
    print("Generating transactions...")
    transactions = [generate_transaction() for _ in range(n_samples)]
    df = pd.DataFrame(transactions)
    
    # Prepare features
    X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']]
    y = df['isFraud']
    
    # Convert type to numeric
    X['type'] = X['type'].map(TYPE_MAPPING)
    
    # Initialize and fit scaler
    print("Fitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    # Save with specific protocol version
    print("Saving model and scaler...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f, protocol=4)
    
    print("Testing saved models...")
    # Verify the saved models can be loaded
    with open('model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        test_scaler = pickle.load(f)
    
    print("Model and scaler saved and verified successfully!")
    print(f"Model file size: {os.path.getsize('model.pkl')} bytes")
    print(f"Scaler file size: {os.path.getsize('scaler.pkl')} bytes")

if __name__ == "__main__":
    main()