"""
Module to handle training, preprocessing and building pipeline
from the dataset
"""

from fastapi import APIRouter
from src.train import train

router = APIRouter()


@router.get("/train")
def train_dataset():
    """
    Endpoint to train the dataset
    """

    print("=" * 60)
    result = train()

    if not result["success"]:
        return {"message": f"Error building pipeline {result["error"]}"}

    return {"message": f"Pipeline built successfully {result["pipeline"]} "}
