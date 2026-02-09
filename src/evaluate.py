"""
Module for evaluating the accuracy and metrics of the trained model
"""

import joblib
from sklearn.metrics import classification_report
from src.load import load_and_split_data
from src.config.settings import DATASET_PATH, MODEL_PATH

pipeline = joblib.load(MODEL_PATH)

X_train, X_test, y_train, y_test = load_and_split_data(DATASET_PATH)

predictions = pipeline.predict(X_test)

model_metrics = classification_report(y_test, predictions)

print(f"Classification Report: {model_metrics}")
