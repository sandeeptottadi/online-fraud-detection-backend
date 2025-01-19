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
    # Generate dataset with more fraudulent cases
    n_samples = 20000  # Increase sample size
    print("Generating transactions...")
    transactions = [generate_transaction() for _ in range(n_samples)]
    df = pd.DataFrame(transactions)
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['isFraud'].value_counts(normalize=True))
    
    # Prepare features
    X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']]
    y = df['isFraud']
    
    # Convert type to numeric
    X['type'] = X['type'].map(TYPE_MAPPING)
    
    # Initialize and fit scaler
    print("\nFitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with better parameters
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,  # Increased from 100
        max_depth=15,      # Increased from 10
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_scaled, y)
    
    # Test model performance
    print("\nTesting model performance...")
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)
    
    # Print some metrics
    from sklearn.metrics import classification_report
    print("\nModel Performance:")
    print(classification_report(y, y_pred))
    
    # Save models
    print("\nSaving models...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Test predictions
    print("\nTesting some cases...")
    test_cases = [
        {
            'step': 210,
            'type': "TRANSFER",
            'amount': 100000.00,
            'oldbalanceOrg': 0.00,
            'oldbalanceDest': 0.00
        },
        {
            'step': 1,
            'type': "PAYMENT",
            'amount': 500.00,
            'oldbalanceOrg': 10000.00,
            'oldbalanceDest': 10000.00
        }
    ]
    
    for case in test_cases:
        input_data = np.array([[
            case['step'],
            TYPE_MAPPING[case['type']],
            case['amount'],
            case['oldbalanceOrg'],
            case['oldbalanceDest']
        ]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0]
        print(f"\nCase: {case}")
        print(f"Fraud probability: {prob[1]:.2%}")

if __name__ == "__main__":
    main()