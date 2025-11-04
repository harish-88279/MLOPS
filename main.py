from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os
import dagshub  # <-- NEW IMPORT

print("Starting API server...")

# --- 1. NEW: Initialize DagsHub tracking ---
try:
    dagshub.init(repo_owner='harish-88279', repo_name='MLOPS', mlflow=True)
    print("DagsHub tracking initialized for API.")
except Exception as e:
    print(f"Error initializing DagsHub: {e}. Model will not load.")
# -------------------------------------------


MODEL_NAME = "iris-classifier"
MODEL_STAGE = "Production" # Fetch the model tagged "Production"

app = FastAPI(title="MLOps Demo API V2")
model = None # Global variable to hold the model

# --- 2. Model Loading Function ---
def load_production_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Attempting to load model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading production model: {e}")
        print("Model will remain 'None'. API will start but /predict will fail.")
        model = None

# --- 3. API Endpoints ---
@app.on_event("startup")
def startup_event():
    # Load the model when the server starts
    load_production_model()

@app.get("/")
def read_root():
    if model:
        return {"message": f"Welcome! The '{MODEL_NAME}' model (stage: {MODEL_STAGE}) is loaded and ready."}
    return {"message": "Welcome! Model is NOT loaded. Check Render logs. Promote a model to 'Production' in MLflow."}

# This is the data schema for a prediction request
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width

@app.post("/predict")
def predict_species(features: IrisFeatures):
    if model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}
    
    try:
        data_df = pd.DataFrame([features.dict()])
        prediction_raw = model.predict(data_df)
        prediction_index = int(prediction_raw[0])
        
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        species_name = species_map.get(prediction_index, "Unknown")
        
        return {
            "prediction_index": prediction_index,
            "prediction_species": species_name,
            "input_features": features.dict()
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}