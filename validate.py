import os
import sys
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.tracking import MlflowClient

print("Starting validation script...")

# --- 1. MLflow setup (Same as train.py) ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASS = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_URI or not MLFLOW_USER or not MLFLOW_PASS:
    print("CRITICAL: MLflow credentials not set.")
    sys.exit(1)

mlflow.set_tracking_uri(MLFLOW_URI)
print(f"Connected to remote MLflow server: {MLFLOW_URI}")

MODEL_NAME = "iris-classifier"

# --- 2. Create a "Golden" Validation Set ---
# We use a DIFFERENT random_state than train.py to ensure
# this data has not been seen by the model.
print("Loading golden validation dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

# We split but only use the 'test' set for validation
# Using random_state=101 to make it different and consistent
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=101)


# --- 3. Get Model Versions ---
try:
    client = MlflowClient()
    
    # Get all versions, sorted from newest to oldest
    versions = client.get_latest_versions(name=MODEL_NAME)
    
    if len(versions) < 2:
        print("Only one model version exists. No old model to compare against. Skipping validation.")
        print("Allowing deployment.")
        sys.exit(0) # Exit successfully

    # The model we just trained
    new_model_version = versions[0] 
    # The "champion" model (the one before it)
    old_model_version = versions[1] 

    print(f"New 'Challenger' Model: Version {new_model_version.version}")
    print(f"Old 'Champion' Model: Version {old_model_version.version}")

except Exception as e:
    print(f"Error getting model versions: {e}")
    sys.exit(1)


# --- 4. Load Models and Compare ---
try:
    # Load new "challenger" model
    new_model_uri = f"models:/{MODEL_NAME}/{new_model_version.version}"
    new_model = mlflow.pyfunc.load_model(new_model_uri)
    
    # Load old "champion" model
    old_model_uri = f"models:/{MODEL_NAME}/{old_model_version.version}"
    old_model = mlflow.pyfunc.load_model(old_model_uri)

    # Evaluate both on the same golden dataset
    new_preds = new_model.predict(X_val)
    old_preds = old_model.predict(X_val)
    
    new_accuracy = accuracy_score(y_val, new_preds)
    old_accuracy = accuracy_score(y_val, old_preds)

    print(f"New Model Accuracy: {new_accuracy}")
    print(f"Old Model Accuracy: {old_accuracy}")

    # --- 5. The Quality Gate ---
    if new_accuracy >= old_accuracy:
        print("Validation PASSED: New model is better or equal to the old model.")
        sys.exit(0) # Exit successfully
    else:
        print(f"Validation FAILED: New model ({new_accuracy}) is worse than old model ({old_accuracy}).")
        print("Deployment will be CANCELED.")
        sys.exit(1) # Exit with an error code

except Exception as e:
    print(f"Error during validation: {e}")
    sys.exit(1)