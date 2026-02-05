"""
Prediction module for titanic survivors
"""

import joblib
import pandas as pd

pipeline = joblib.load("models/titanic_pipeline.pkl")


def predict(data):
    """
    Predict if a person survived on the titanic
    given the input data
    """

    df = pd.DataFrame([data])
    prediction = pipeline.predict(df)

    return int(prediction[0])
