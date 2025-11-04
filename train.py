import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

print("Starting training script...")

# --- 1. MLflow setup (This part is correct) ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASS = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_URI or not MLFLOW_USER or not MLFLOW_PASS:
    print("CRITICAL: MLflow credentials not set.")
    exit(1)

mlflow.set_tracking_uri(MLFLOW_URI)
print(f"Logging to remote MLflow server: {MLFLOW_URI}")
# -----------------------------------------------

# --- 2. Run the experiment ---
mlflow.set_experiment("Iris-Classifier-Demo")

with mlflow.start_run() as run:
    print(f"Starting run: {run.info.run_id}")

    # Load and train
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_iter_param = 200
    model = LogisticRegression(max_iter=max_iter_param)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # Log params and metrics
    print(f"Logging metric: accuracy={accuracy}")
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy)

    # --- 3. THIS IS THE MODIFIED BLOCK ---
    print("Logging model artifact...")
    # Step 1: Log the model *without* registering it
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5)
        # We removed 'registered_model_name' from here
    )

    # Step 2: Register the model explicitly
    print("Registering the model...")
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name="iris-classifier"
    )
    print("Model registered successfully.")
    # -------------------------------------

    print(f"Run {run.info.run_id} finished.")

print("Training script finished successfully.")