# Potato Disease Classifier

This repository contains a FastAPI application that classifies **Potato Leaf** images into three categories:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

It includes two TensorFlow/Keras models:

1. **Base Model (1)**
2. **Optimized Model (2)**

## Overview

- **Goal**: Quickly diagnose potato leaf diseases (or confirm health) using deep learning.
- **Models**: Trained with TensorFlow. The “Base Model” is simpler; the “Optimized Model” applies additional tuning.
- **Frontend**: A simple HTML/JS interface (`index.html`) for uploading images and displaying predictions.

## Dataset

The training data is from [this Kaggle dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset).  
Please visit the link to download the images if you’d like to replicate or retrain the models.

**Note**: We do **not** include the raw dataset in this repo to keep the project size manageable.

## Screenshots

Inside the `screenshots/` folder, you’ll find images illustrating the application in action:

| Base Model (1)                             | Optimized Model (2)                            |
|--------------------------------------------|-----------------------------------------------|
| ![Base Model Screenshot](screenshots/base_model.png) | ![Optimized Model Screenshot](screenshots/optimized_model.png) |

For example:

![Base Model Prediction](screenshots/Screenshot2025-03-05_083655.png)
![Optimized Model Prediction](screenshots/Screenshot2025-03-05_083549.png)

These show how the app looks when classifying a healthy potato leaf.

## Live Demo

A live version is deployed on **Google Cloud Run** at:  
[https://potato-api-443221614300.us-central1.run.app/](https://potato-api-443221614300.us-central1.run.app/)

Upload an image, select a model version, and see classification results.

## Project Structure

├── api/ 
│    └── main.py
│    ├── frontend/
│          └── index.html
├── models/ 
│    └── base.h5 
│    └── optimized.h5 
├── Dockerfile 
├── requirements.txt 
├── training
│    └── training_base.ipynb 
│    └── training_optimized.ipynb
├── README.md

├── api/
│   ├── main.py
├── models/
│   ├── base.h5
│   └── optimized.h5
├── screenshots/
│   ├── Screenshot2025-03-05_083655.png
│   └── Screenshot2025-03-05_083549.png
├── Dockerfile
├── index.html
├── requirements.txt
├── training_base.ipynb
├── training_optimized.ipynb
└── README.md




- **api/main.py**: FastAPI logic (endpoints for image upload & inference).
- **models/**: Folder with pre-trained `.h5` files (base & optimized).
- **screenshots/**: UI screenshots.
- **index.html**: Frontend for uploading images.
- **training_base.ipynb**, **training_optimized.ipynb**: Jupyter notebooks to train each model.
- **requirements.txt**: Python dependencies (TensorFlow, FastAPI, etc.).

## Running Locally

1. **Clone** this repo:
   ```bash
   git clone https://github.com/hussinxx700/potato-disease-classifier.git
   cd potato-disease-classifier

2. Create a Python virtual environment **(recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows

1. Install Dependencies:
   ```bash
   pip install -r requirements.txt

1. Start the FastAPI App:
   ```bash
   python api/main.py

By default, it listens on 127.0.0.1:8080.
Open http://127.0.0.1:8080 in your browser to view the interface.

## Docker Usage

1. Build the Docker image
   ```bash
   docker build -t potato_api:v1 .

2. Run the container, mapping port 8080
   ```bash
    docker run -d -p 8080:8080 potato_api:v1

## Google Cloud Run Deployment

1. Build with Cloud Build
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/potato_api:v1 .

2. Deploy
   ```bash
   gcloud run deploy potato-api \
     --image gcr.io/<PROJECT_ID>/potato_api:v1 \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
