from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os
# --- NO "import dagshub" ---

print("Starting API server...")

# --- 1. REVERT to the original MLflow setup ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"MLflow tracking URI set to: {MLFLOW_URI}")
else:
    print("CRITICAL: MLFLOW_TRACKING_URI not set. Model will not load.")
# -----------------------------------------------

# --- 2. The rest is the same ---
MODEL_NAME = "iris-classifier"
MODEL_STAGE = "Production"
# ... (all the API code remains unchanged)
app = FastAPI(title="MLOps Demo API V2")
model = None 
def load_production_model():
    global model
    if not MLFLOW_URI:
        print("MLFLOW_URI not set, cannot load model.")
        return
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"Attempting to load model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading production model: {e}")
        model = None
@app.on_event("startup")
def startup_event():
    load_production_model()
@app.get("/")
def read_root():
    if model:
        return {"message": f"Welcome! The '{MODEL_NAME}' model (stage: {MODEL_STAGE}) is loaded and ready."}
    return {"message": "Welcome! Model is NOT loaded. Check Render logs. Promote a model to 'Production' in MLflow."}
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
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
        return {"prediction_index": prediction_index, "prediction_species": species_name}
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}