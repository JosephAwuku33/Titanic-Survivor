"""
API module for titanic survivor prediction
"""

from fastapi import FastAPI
from .router.predictions import predictions
from .router.training import train


app = FastAPI()

app.title = "Titanic Survivor Predictor"
app.include_router(predictions.router)
app.include_router(train.router)


@app.get("/")
def start():
    """
    The default endpoint
    """
    return {"message": "Titanic Survivor Predictor is active"}
