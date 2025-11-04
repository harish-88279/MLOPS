import os
import sys
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.tracking import MlflowClient

print("Starting validation script...")

# --- 1. MLflow setup (No change) ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASS = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_URI or not MLFLOW_USER or not MLFLOW_PASS:
    print("CRITICAL: MLflow credentials not set.")
    sys.exit(1)

mlflow.set_tracking_uri(MLFLOW_URI)
print(f"Connected to remote MLflow server: {MLFLOW_URI}")

MODEL_NAME = "iris-classifier"

# --- 2. Golden Validation Set (No change) ---
print("Loading golden validation dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=101)


# --- 3. THIS BLOCK IS NEW ---
print("Getting model versions from file...")
try:
    # Read the new version number created by train.py
    with open("version.txt", "r") as f:
        new_version_str = f.read()
    
    new_version = int(new_version_str)
    old_version = new_version - 1

    if old_version < 1:
        print(f"New model is Version {new_version}. This is the first model.")
        print("No old model to compare against. Skipping validation.")
        print("Allowing deployment.")
        sys.exit(0) # Exit successfully

    print(f"New 'Challenger' Model: Version {new_version}")
    print(f"Old 'Champion' Model: Version {old_version}")

except Exception as e:
    print(f"Error reading version.txt: {e}")
    sys.exit(1)
# -----------------------------


# --- 4. Load Models and Compare ---
try:
    # Load new "challenger" model by its *exact* version
    new_model_uri = f"models:/{MODEL_NAME}/{new_version}"
    new_model = mlflow.pyfunc.load_model(new_model_uri)
    
    # Load old "champion" model by its *exact* version
    old_model_uri = f"models:/{MODEL_NAME}/{old_version}"
    old_model = mlflow.pyfunc.load_model(old_model_uri)

    # Evaluate both
    new_preds = new_model.predict(X_val)
    old_preds = old_model.predict(X_val)
    
    new_accuracy = accuracy_score(y_val, new_preds)
    old_accuracy = accuracy_score(y_val, old_preds)

    print(f"New Model Accuracy: {new_accuracy}")
    print(f"Old Model Accuracy: {old_accuracy}")

    # --- 5. The Quality Gate (No change) ---
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