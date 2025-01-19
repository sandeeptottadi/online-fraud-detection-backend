import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys

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

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    try:
        # Save with protocol=4 for better compatibility
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        print(f"Model saved successfully to {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f, protocol=4)
        print(f"Scaler saved successfully to {scaler_path}")
        
        # Verify the files can be loaded
        with open(model_path, 'rb') as f:
            test_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            test_scaler = pickle.load(f)
        print("Verified: Model and scaler can be loaded successfully")
        
    except Exception as e:
        print(f"Error saving model or scaler: {str(e)}")
        sys.exit(1)

def main():
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"scikit-learn version: {pd.__version__}")
    
    # Generate dataset
    n_samples = 10000
    print(f"\nGenerating {n_samples} transactions...")
    transactions = [generate_transaction() for _ in range(n_samples)]
    df = pd.DataFrame(transactions)
    
    # Prepare features
    X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']]
    y = df['isFraud']
    
    # Convert type to numeric
    X['type'] = X['type'].map(TYPE_MAPPING)
    
    # Initialize and fit scaler
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_scaled, y)
    
    # Save models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model.pkl')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    save_model_and_scaler(model, scaler, model_path, scaler_path)
    
    # Print statistics
    print("\nModel Training Summary:")
    print("-" * 50)
    print(f"Total Transactions: {len(df)}")
    print(f"Fraudulent Transactions: {sum(df['isFraud'])}")
    print(f"Legitimate Transactions: {len(df) - sum(df['isFraud'])}")
    
    # Test predictions
    test_cases = [
        {
            'step': 1,
            'type': "PAYMENT",
            'amount': 50000.00,
            'oldbalanceOrg': 200000.00,
            'oldbalanceDest': 100000.00
        },
        {
            'step': 210,
            'type': "TRANSFER",
            'amount': 5.00,
            'oldbalanceOrg': 0.00,
            'oldbalanceDest': 0.00
        }
    ]
    
    print("\nTest Predictions:")
    print("-" * 50)
    for case in test_cases:
        input_data = np.array([[
            case['step'],
            TYPE_MAPPING[case['type']],
            case['amount'],
            case['oldbalanceOrg'],
            case['oldbalanceDest']
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        print(f"\nTransaction:")
        print(f"Type: {case['type']}")
        print(f"Amount: ${case['amount']}")
        print(f"Prediction: {'Fraudulent' if prediction[0] else 'Legitimate'}")

if __name__ == "__main__":
    main()