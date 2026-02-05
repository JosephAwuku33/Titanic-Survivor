"""
Module for evaluating the accuracy and metrics of the trained model
"""

from pathlib import Path
import joblib
from sklearn.metrics import classification_report
from src.load import load_and_split_data

pipeline = joblib.load("models/titanic_pipeline.pkl")

BASE_DIR = Path("data/raw") / "titanic.csv"
X_train, X_test, y_train, y_test = load_and_split_data(BASE_DIR)

predictions = pipeline.predict(X_test)

model_metrics = classification_report(y_test, predictions)

print(f"Classification Report: {model_metrics}")
