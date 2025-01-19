import pickle
import os
import sys  # Added sys import
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(
    title="Online Fraud Detection API",
    description="API for detecting fraudulent online transactions",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None

# Load models on startup
def load_models():
    global model, scaler
    try:
        # List directory contents for debugging
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
        
        # Load model
        print("Attempting to load model...")
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully")
        
        # Load scaler
        print("Attempting to load scaler...")
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        print("Scaler loaded successfully")
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Python version: {sys.version}")
        if model is None:
            print("Model failed to load")
        if scaler is None:
            print("Scaler failed to load")
        raise Exception(f"Error loading models: {str(e)}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
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
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model or scaler not loaded. Please check server logs."
        )
    
    try:
        print(f"Received prediction request for data: {data}")
        # Convert input to model-ready format
        input_data = np.array([[
            data.step,
            TYPE_MAPPING[data.type],
            data.amount,
            data.oldbalanceOrg,
            data.oldbalanceDest
        ]])
        
        print(f"Transformed input data shape: {input_data.shape}")
        
        # Transform using pre-fitted scaler
        input_scaled = scaler.transform(input_data)
        print(f"Scaled input data shape: {input_scaled.shape}")
        
        # Make prediction
        prediction = model.predict(input_scaled)
        print(f"Prediction result: {prediction[0]}")
        
        return {
            "fraud_prediction": bool(prediction[0]),
            "success": True,
            "message": "Fraudulent transaction detected!" if prediction[0] else "Transaction appears legitimate"
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "scaler_status": scaler_status,
        "current_directory": os.getcwd(),
        "directory_contents": os.listdir()
    }

# Load models when the application starts
load_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)