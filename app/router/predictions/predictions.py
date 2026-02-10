"""
Module to handle survivor prediction route, endpoint
"""

from fastapi import APIRouter
from app.schema.titanic_data import SurvivorInput
from src.predict import predict

router = APIRouter()


@router.post("/predict")
def make_prediction(data: SurvivorInput):
    """
    Endpoint to get the if a person survived the titanic
    """
    print(f"This is the data {data}")
    result = predict(data.dict())
    print(f"This is the result though {result}")
    prediction = "yes" if int(result) == 1 else "no"
    return {"Did the person most likely survive": prediction}
