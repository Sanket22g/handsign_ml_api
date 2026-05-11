import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Define absolute paths based on the current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(filename: str) -> str:
    # First check if the file is in the "model" folder
    if os.path.exists(os.path.join(BASE_DIR, "model", filename)):
        return os.path.join(BASE_DIR, "model", filename)
    # Otherwise, assume it's in the same folder as main.py
    return os.path.join(BASE_DIR, filename)

MODEL_PATH = get_model_path("gesture_model_isl30.pkl")
SCALER_PATH = get_model_path("scaler_isl30.pkl")
LE_PATH = get_model_path("label_encoder_isl30.pkl")

# Global dictionary to store the loaded ML artifacts
ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the machine learning model, scaler, and label encoder
    efficiently when the FastAPI server starts.
    """
    try:
        # Use joblib instead of pickle for scikit-learn models to avoid UnpicklingError
        ml_artifacts["model"] = joblib.load(MODEL_PATH)
        ml_artifacts["scaler"] = joblib.load(SCALER_PATH)
        ml_artifacts["label_encoder"] = joblib.load(LE_PATH)
        print("✅ Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ Failed to find model file: {e}")
        raise e
    except Exception as e:
        print(f"❌ Error loading ML artifacts: {e}")
        raise e
    
    # Server is running and ready to handle requests
    yield
    
    # Clean up (Optional) when the server shuts down
    ml_artifacts.clear()
    print("🧹 Cleaned up ML artifacts.")

# Initialize the FastAPI application with the lifespan context manager
app = FastAPI(title="ISL Smart Glove Predictor", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse

@app.get("/", include_in_schema=False)
async def root():
    """Serve the simple UI index.html"""
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Define the Pydantic models for request and response validation
class PredictRequest(BaseModel):
    # Enforcing exactly 22 floating point features
    features: List[float] = Field(
        ..., 
        min_length=22, 
        max_length=22, 
        description="Exactly 22 floating-point features from the smart glove."
    )
    word_buffer: Optional[List[str]] = Field(
        default_factory=list, 
        description="Optional list to keep track of a sentence."
    )

class PredictResponse(BaseModel):
    predicted_word: str
    word_buffer: List[str]

@app.post("/predict", response_model=PredictResponse)
async def predict_gesture(request: PredictRequest):
    """
    Accepts 22 sensor features, scales them, and predicts the ISL gesture.
    It returns the predicted word and an updated word buffer.
    """
    if not ml_artifacts:
        raise HTTPException(status_code=503, detail="Models are not loaded on the server.")

    try:
        # 1. Reshape the input features to a 2D array: (1, 22)
        features_array = np.array(request.features).reshape(1, -1)
        
        # 2. Transform the features using the loaded scaler
        scaler = ml_artifacts["scaler"]
        scaled_features = scaler.transform(features_array)
        
        # 3. Make a prediction
        model = ml_artifacts["model"]
        prediction = model.predict(scaled_features)
        
        # 4. Inverse-transform the prediction using the label encoder
        label_encoder = ml_artifacts["label_encoder"]
        predicted_word = label_encoder.inverse_transform(prediction)[0]
        
        # 5. Update the word buffer
        buffer = request.word_buffer.copy() if request.word_buffer is not None else []
        buffer.append(str(predicted_word))
        
        return PredictResponse(
            predicted_word=str(predicted_word),
            word_buffer=buffer
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data processing error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error during prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    # Allow running the server locally with `python main.py`
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
