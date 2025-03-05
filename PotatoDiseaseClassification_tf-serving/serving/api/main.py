import requests
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model versions and their corresponding TensorFlow Serving ports
MODEL_VERSIONS = {
    "1": "base",
    "2": "transfer_learning",
    "3": "optimized",
}

# Class names for predictions
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def preprocess_image(image_byte):
    """
    Reads an image, converts it to NumPy, and ensures correct format (1, 256, 256, 3).
    """
    try:
        nparr = np.frombuffer(image_byte, np.uint8)

        # Decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image if necessary
        image = cv2.resize(image, (256, 256))  # Ensuring correct size
        image = np.expand_dims(image, axis=0)  # Model expects batch dimension

        return image.tolist()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

#apply softmax 
def softmax (raw_predictions):

    predictions = np.exp(raw_predictions) / np.sum(np.exp(raw_predictions))
    return predictions 


# Define API Response Format
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict


@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    version: str = Query("3", description="Model version: 1, 2, or 3")
):
    """
    Receives an image, preprocesses it, sends it to the selected TensorFlow Serving model, and processes the output.
    """

    # Validate model version
    if version not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail="Invalid model version. Choose from 1, 2, or 3.")

    # Get TensorFlow Serving URL for the selected model version
    
    TF_SERVING_URL = f"http://host.docker.internal:8501/v1/models/potato_classifier/versions/{version}:predict"

    try:
        # Preprocess image
        processed_image = preprocess_image(await file.read())

        # Send request to TensorFlow Serving
        response = requests.post(TF_SERVING_URL, json={'instances': processed_image})

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail='Error from TensorFlow Serving')

        # Process the model output
        predictions = response.json()["predictions"][0]

        if version != '2':

            #Apply softmax on raw pridictions for (base) and (optimiaed) models
            predictions = softmax(predictions)
            
            
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]

        prob_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

        return {
            "predicted_class": predicted_class,
            "confidence": float(predictions[predicted_index]),
            "probabilities": prob_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the frontend folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")
