import pickle
import os
import sys
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
import joblib  # Add this import

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
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir())
        
        # Try different loading methods
        try:
            print("Attempting to load model with pickle...")
            with open("model.pkl", "rb") as file:
                model = pickle.load(file)
        except Exception as e1:
            print(f"Pickle load failed: {str(e1)}")
            print("Attempting to load model with joblib...")
            try:
                model = joblib.load("model.pkl")
            except Exception as e2:
                print(f"Joblib load failed: {str(e2)}")
                raise Exception("Could not load model with either method")

        try:
            print("Attempting to load scaler with pickle...")
            with open("scaler.pkl", "rb") as file:
                scaler = pickle.load(file)
        except Exception as e1:
            print(f"Pickle load failed: {str(e1)}")
            print("Attempting to load scaler with joblib...")
            try:
                scaler = joblib.load("scaler.pkl")
            except Exception as e2:
                print(f"Joblib load failed: {str(e2)}")
                raise Exception("Could not load scaler with either method")

        print("Model and scaler loaded successfully")
        return True
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Python version: {sys.version}")
        if model is None:
            print("Model failed to load")
        if scaler is None:
            print("Scaler failed to load")
        return False

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        json_schema_extra = {  # Updated from schema_extra
            "example": {
                "step": 1,
                "type": "PAYMENT",
                "amount": 9839.64,
                "oldbalanceOrg": 170136.0,
                "oldbalanceDest": 0.0
            }
        }

TYPE_MAPPING = {
    "PAYMENT": 0,
    "CASH_IN": 1,
    "DEBIT": 2,
    "CASH_OUT": 3,
    "TRANSFER": 4
}

@app.get("/")
async def read_root():
    return {
        "message": "Online Fraud Detection API",
        "docs_url": "/docs",
        "health": "OK"
    }

@app.post("/predict")
async def predict(data: FraudInput):
    if model is None or scaler is None:
        # Try loading models again
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail="Model service not ready. Please try again later."
            )
    
    try:
        print(f"Received prediction request for data: {data}")
        input_data = np.array([[
            data.step,
            TYPE_MAPPING[data.type],
            data.amount,
            data.oldbalanceOrg,
            data.oldbalanceDest
        ]])
        
        print(f"Transformed input data shape: {input_data.shape}")
        input_scaled = scaler.transform(input_data)
        print(f"Scaled input data shape: {input_scaled.shape}")
        
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

@app.get("/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "scaler_status": scaler_status,
        "current_directory": os.getcwd(),
        "directory_contents": os.listdir(),
        "python_version": sys.version
    }

# Try to load models on startup, but don't fail if they don't load
load_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)