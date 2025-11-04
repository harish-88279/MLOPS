import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
# --- NO "import dagshub" ---

print("Starting training script...")

# --- 1. REVERT to the original MLflow setup ---
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASS = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_URI or not MLFLOW_USER or not MLFLOW_PASS:
    print("CRITICAL: MLflow credentials not set.")
    exit(1)

mlflow.set_tracking_uri(MLFLOW_URI)
print(f"Logging to remote MLflow server: {MLFLOW_URI}")
# -----------------------------------------------

# --- 2. The rest is the same ---
mlflow.set_experiment("Iris-Classifier-Demo")

with mlflow.start_run() as run:
    # ... (all the training code remains unchanged)
    print(f"Starting run: {run.info.run_id}")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_iter_param = 200
    model = LogisticRegression(max_iter=max_iter_param)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Logging metric: accuracy={accuracy}")
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy)
    print("Logging and registering the model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5),
        registered_model_name="iris-classifier"
    )
    print(f"Run {run.info.run_id} finished.")

print("Training script finished successfully.")