import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Type mapping - moved to top for reuse
TYPE_MAPPING = {
    "PAYMENT": 0,
    "CASH_IN": 1,
    "DEBIT": 2,
    "CASH_OUT": 3,
    "TRANSFER": 4
}

# Create more realistic fraud patterns
def generate_transaction():
    is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])
    
    if is_fraud:
        # Fraudulent patterns
        step = np.random.randint(200, 220)  # Suspicious time window
        type = np.random.choice(['TRANSFER', 'CASH_OUT'], p=[0.7, 0.3])
        amount = np.random.choice([
            np.random.uniform(0.1, 10.0),  # Very small amounts
            np.random.uniform(50000, 1000000)  # Very large amounts
        ], p=[0.6, 0.4])
        oldbalanceOrg = np.random.choice([
            0.0,  # Zero balance
            amount * 0.1,  # Insufficient balance
            amount * 0.5
        ], p=[0.4, 0.4, 0.2])
        oldbalanceDest = np.random.choice([
            0.0,
            amount * 0.1
        ], p=[0.8, 0.2])
    else:
        # Legitimate patterns
        step = np.random.randint(1, 500)
        type = np.random.choice(['PAYMENT', 'CASH_IN', 'DEBIT', 'TRANSFER', 'CASH_OUT'])
        amount = np.random.uniform(100, 50000)
        oldbalanceOrg = amount * np.random.uniform(2, 5)  # Sufficient balance
        oldbalanceDest = np.random.uniform(1000, 100000)
    
    return {
        'step': step,
        'type': type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'oldbalanceDest': oldbalanceDest,
        'isFraud': is_fraud
    }

# Generate dataset
n_samples = 10000
transactions = [generate_transaction() for _ in range(n_samples)]
df = pd.DataFrame(transactions)

# Prepare features
X = df[['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']]
y = df['isFraud']

# Convert type to numeric
X['type'] = X['type'].map(TYPE_MAPPING)

# Fit scaler on training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model with better parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)
model.fit(X_scaled, y)

# Save both model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Print some statistics
print("\nModel Training Summary:")
print("-" * 50)
print(f"Total Transactions: {len(df)}")
print(f"Fraudulent Transactions: {sum(df['isFraud'])}")
print(f"Legitimate Transactions: {len(df) - sum(df['isFraud'])}")
print("\nModel and scaler saved successfully!")

# Test some cases
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
        TYPE_MAPPING[case['type']],  # Convert type to numeric
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