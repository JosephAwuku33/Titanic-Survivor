"""
Module to export the function which loads the data,
set's up the features, and separate the data into
training and testing features
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


def load_and_split_data(path: Path):
    """
    Function that takes in a path to the dataset as a param
    separates the features into classes, as well as the target,
    and then splits the data into its training and testing sections
    """

    df = pd.read_csv(path)

    # define X related values
    feature_columns = ["Age", "Fare", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
    x = df[feature_columns]

    # define the target y
    y = df["Survived"]

    # split the data into its testing and training sections
    return train_test_split(x, y, test_size=0.2, random_state=42)
