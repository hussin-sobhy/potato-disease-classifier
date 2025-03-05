import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; restrict in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    model_base = tf.keras.models.load_model("../models/base.h5")
    model_optimized = tf.keras.models.load_model("../models/optimized.h5")
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")


LOADED_MODELS = {
    "1": model_base,
    "2": model_optimized,
}

# Class names for predictions
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess_image(image_bytes: bytes):
    """
    Reads an image from raw bytes, decodes with OpenCV,
    and returns shape (1, 256, 256, 3).
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def softmax(raw_predictions: np.ndarray) -> np.ndarray:
    """
    Apply softmax to raw logits (if the model doesn't have final softmax).
    """
    exp_preds = np.exp(raw_predictions)
    return exp_preds / np.sum(exp_preds)

# Response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict

@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    version: str = Query("1", description="Model version: 1 or 2")
):
    """
    Receives an image, preprocesses it, runs inference with the selected model,
    returns predicted class & probabilities.
    """

    # Validate version
    if version not in LOADED_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model version. Choose '1' or '2'.")

    # Preprocess image
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)

    
    model = LOADED_MODELS[version]

    # Run inference
    raw_output = model.predict(processed_image)  
    predictions = raw_output[0]  

    # Apply softmax
    predictions = softmax(predictions)

    
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]

    # Build probability dict
    prob_dict = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": float(predictions[predicted_index]),
        "probabilities": prob_dict
    }

# Serve the frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")
