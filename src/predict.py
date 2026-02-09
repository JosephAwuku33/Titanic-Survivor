"""
Prediction module for titanic survivors
"""

import joblib
import pandas as pd
from src.config.settings import MODEL_PATH

pipeline = joblib.load(MODEL_PATH)


def predict(data):
    """
    Predict if a person survived on the titanic
    given the input data
    """

    df = pd.DataFrame([data])
    prediction = pipeline.predict(df)

    return int(prediction[0])
