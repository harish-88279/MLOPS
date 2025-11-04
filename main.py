from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient

print("Starting API server...")

# --- 1. MLflow setup (This part is correct) ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f"MLflow tracking URI set to: {MLFLOW_URI}")
else:
    print("CRITICAL: MLFLOW_TRACKING_URI not set. Model will not load.")
# -----------------------------------------------

MODEL_NAME = "iris-classifier"
app = FastAPI(title="MLOps Demo API V2")
model = None 

def load_production_model():
    global model
    if not MLFLOW_URI:
        print("MLFLOW_URI not set, cannot load model.")
        return
    try:
        client = MlflowClient()
        print(f"Searching for latest version of model: {MODEL_NAME}")
        latest_versions = client.get_latest_versions(name=MODEL_NAME)
        if not latest_versions:
             print(f"Error: No versions found for model '{MODEL_NAME}'.")
             model = None
             return
        latest_version = latest_versions[0]
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        print(f"Attempting to load model from: {model_uri} (Version: {latest_version.version})")
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
        return {"message": f"Welcome! The latest version of '{MODEL_NAME}' is loaded and ready."}
    return {"message": "Welcome! Model is NOT loaded. Check Render logs."}

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- THIS FUNCTION IS NOW FIXED ---
@app.post("/predict")
def predict_species(features: IrisFeatures):
    if model is None:
        return {"error": "Model is not loaded. Cannot make predictions."}
    
    try:
        # 1. Get the data from the API request
        input_data = features.dict()

        # 2. Rename keys to match the training data schema
        #    This is the crucial fix
        renamed_data = {
            "sepal length (cm)": input_data['sepal_length'],
            "sepal width (cm)": input_data['sepal_width'],
            "petal length (cm)": input_data['petal_length'],
            "petal width (cm)": input_data['petal_width']
        }

        # 3. Create the DataFrame with the *correct* column names
        data_df = pd.DataFrame([renamed_data])
        
        # 4. Predict
        prediction_raw = model.predict(data_df)
        prediction_index = int(prediction_raw[0])
        
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        species_name = species_map.get(prediction_index, "Unknown")
        
        return {
            "prediction_index": prediction_index,
            "prediction_species": species_name
        }
    except Exception as e:
        # Return the actual error to help with debugging
        return {"error": f"Prediction error: {str(e)}"}