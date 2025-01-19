import pickle
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware

# Load the model and scaler
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise Exception("Model or scaler file not found")

# Initialize FastAPI
app = FastAPI(
    title="Online Fraud Detection API",
    description="API for detecting fraudulent online transactions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class FraudInput(BaseModel):
    step: int = 1
    type: Literal["PAYMENT", "CASH_IN", "DEBIT", "CASH_OUT", "TRANSFER"]
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float

    @validator('amount', 'oldbalanceOrg', 'oldbalanceDest')
    def validate_amounts(cls, v):
        if not np.isfinite(v):
            raise ValueError('Amount must be a finite number')
        if v < 0:
            raise ValueError('Amount cannot be negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "oldbalanceOrg": 170136.0,
                "oldbalanceDest": 0.0
            }
        }

# Define type mapping
TYPE_MAPPING = {
    "PAYMENT": 0,
    "CASH_IN": 1,
    "DEBIT": 2,
    "CASH_OUT": 3,
    "TRANSFER": 4
}

# Define the root route
@app.get("/")
async def read_root():
    return {
        "message": "Online Fraud Detection API",
        "docs_url": "/docs",
        "health": "OK"
    }

# Endpoint for prediction
@app.post("/predict")
async def predict(data: FraudInput):
    try:
        print(data)
        # Convert input to model-ready format
        input_data = np.array([[
            data.step,
            TYPE_MAPPING[data.type],
            data.amount,
            data.oldbalanceOrg,
            data.oldbalanceDest
        ]])
        
        # Transform using pre-fitted scaler
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        return {
            "fraud_prediction": bool(prediction[0]),
            "success": True,
            "message": "Fraudulent transaction detected!" if prediction[0] else "Transaction appears legitimate"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

