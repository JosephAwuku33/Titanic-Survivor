"""
Module for handling the schema of the titanic-data
"""

from typing import Literal
from pydantic import BaseModel


class SurvivorInput(BaseModel):
    """
    Input data validation model for the predictor
    """

    Age: int
    Fare: float
    Sex: Literal["male", "female"]
    Pclass: Literal[1, 2, 3]
    Embarked: Literal["Q", "C", "S"]
    SibSp: int
    Parch: int
