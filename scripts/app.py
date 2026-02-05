from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Dict
import json
from datetime import datetime

# Load the trained model
MODEL_PATH = os.getenv('MODEL_PATH', 'models-data/turbofan_model_v1.joblib')

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load feature names (from training)
try:
    feature_importance_path = MODEL_PATH.replace('.joblib', '_feature_importance.csv')
    feature_df = pd.read_csv(feature_importance_path)
    FEATURE_COLUMNS = feature_df['feature'].tolist()
    print(f"✅ Loaded {len(FEATURE_COLUMNS)} feature names")
except:
    # Default feature columns if file not found
    FEATURE_COLUMNS = [
        'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05',
        'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
        'sensor_21'
    ]
    print(f"⚠ Using default feature columns ({len(FEATURE_COLUMNS)} features)")

# Create FastAPI app
app = FastAPI(
    title="NASA Turbofan Predictive Maintenance API",
    description="API for predicting engine failure in NASA Turbofan dataset",
    version="1.0.0",
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, 0.3, 0.2] + [0.1] * 21  # 24 features total
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    engine_status: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    features_count: int
    timestamp: str

class BatchPredictionRequest(BaseModel):
    engine_readings: List[List[float]]

class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "none",
        "features_count": len(FEATURE_COLUMNS),
        "timestamp": datetime.utcnow().isoformat()
    }

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "model_params": model.get_params(),
        "features": FEATURE_COLUMNS,
        "features_count": len(FEATURE_COLUMNS),
        "n_classes": model.n_classes_ if hasattr(model, 'n_classes_') else None,
        "n_features_in": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
    }
    return info

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction for engine failure"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input length
    if len(request.features) != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(FEATURE_COLUMNS)} features, got {len(request.features)}"
        )
    
    try:
        # Convert to numpy array and reshape for prediction
        features_array = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]  # Probability of failure
        
        # Determine engine status
        engine_status = "CRITICAL - Failure predicted" if prediction == 1 else "NORMAL"
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "engine_status": engine_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make predictions for multiple engine readings"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for i, features in enumerate(request.engine_readings):
            # Validate input length
            if len(features) != len(FEATURE_COLUMNS):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Row {i}: Expected {len(FEATURE_COLUMNS)} features, got {len(features)}"
                )
            
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0][1]
            
            results.append({
                "engine_id": i,
                "prediction": int(prediction),
                "probability": float(probability),
                "engine_status": "CRITICAL - Failure predicted" if prediction == 1 else "NORMAL"
            })
        
        return {
            "predictions": results,
            "total_predictions": len(results),
            "critical_count": sum(1 for r in results if r["prediction"] == 1),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NASA Turbofan Predictive Maintenance API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)