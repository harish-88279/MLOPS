import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub  # <-- NEW IMPORT

print("Starting training script...")

# --- 1. NEW: Initialize DagsHub tracking ---
# This automatically finds your repo and sets up MLflow
# Make sure your repo_owner and repo_name are correct!
try:
    dagshub.init(repo_owner='harish-88279', repo_name='MLOPS', mlflow=True)
    print("DagsHub tracking initialized successfully.")
except Exception as e:
    print(f"Error initializing DagsHub: {e}. Exiting.")
    exit(1)
# -------------------------------------------

# --- 2. Run the experiment ---
mlflow.set_experiment("Iris-Classifier-Demo")

with mlflow.start_run() as run:
    print(f"Starting run: {run.info.run_id}")

    # Load Data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    max_iter_param = 200
    model = LogisticRegression(max_iter=max_iter_param)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # Log to DagsHub
    print(f"Logging metric: accuracy={accuracy}")
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy)

    # Log and Register the Model
    print("Logging and registering the model...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train.head(5),
        # This gives the model a name for the API to find
        registered_model_name="iris-classifier"
    )

    print(f"Run {run.info.run_id} finished.")

print("Training script finished successfully.")