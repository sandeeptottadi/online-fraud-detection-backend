import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.pkl not found!")
    exit(1)

# Test cases
test_cases = [
    {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "oldbalanceOrg": 170136.0,
        "oldbalanceDest": 10000.0
    },
    {
        "step": 50,
        "type": "CASH_IN",
        "amount": 20000.00,
        "oldbalanceOrg": 100000.00,
        "oldbalanceDest": 50000.00
    },
    {
        "step": 24,
        "type": "TRANSFER",
        "amount": 15000.00,
        "oldbalanceOrg": 100000.00,
        "oldbalanceDest": 50000.00
    }
]

# Type mapping
TYPE_MAPPING = {
    "PAYMENT": 0,
    "CASH_IN": 1,
    "DEBIT": 2,
    "CASH_OUT": 3,
    "TRANSFER": 4
}

# Create a scaler
scaler = StandardScaler()

# Test each case
print("\nTesting predictions:")
print("-" * 50)

for case in test_cases:
    # Convert input to model-ready format
    input_data = np.array([[
        case["step"],
        TYPE_MAPPING[case["type"]],
        case["amount"],
        case["oldbalanceOrg"],
        case["oldbalanceDest"]
    ]])
    
    # Scale the input
    input_scaled = scaler.fit_transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    print(f"\nTest Case:")
    print(f"Type: {case['type']}")
    print(f"Amount: ${case['amount']}")
    print(f"Sender Balance: ${case['oldbalanceOrg']}")
    print(f"Recipient Balance: ${case['oldbalanceDest']}")
    print(f"Prediction: {'Fraudulent' if prediction[0] else 'Legitimate'}")

print("\nDone testing!") 