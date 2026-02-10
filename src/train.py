"""
Module for handling preprocessing of data and training of the model.

This script loads the Titanic dataset, creates preprocessing pipelines for
different feature groups, combines them into a single preprocessing pipeline,
and trains a LogisticRegression model on the processed data.
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.load import load_and_split_data
from src.utils.transformer import FamilyFeatures
from src.config.settings import DATASET_PATH, MODEL_PATH

# Load training and testing data
X_train, X_test, y_train, y_test = load_and_split_data(DATASET_PATH)


def train() -> dict:
    """
    Trains the Titanic model and returns a status dictionary.

    Returns:
        dict: {
            "success": bool,
            "pipeline": Pipeline object or None,
            "error": str or None
        }
    """
    try:
        # 1. Feature groups
        age_feature = ["Age"]
        numerical_features = ["Fare"]
        categorical_features = ["Embarked", "Sex"]
        passenger_class_feature = ["Pclass"]
        family_features = ["SibSp", "Parch"]

        # 2. Pipelines (same as yours)
        age_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
        pclass_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

        # Note: drop="first" is used here for Logistic Regression
        cat_pipeline = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]
        )

        family_pipeline = Pipeline(
            steps=[
                ("family_features", FamilyFeatures()),
            ]
        )

        # 3. Combine using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("age", age_pipeline, age_feature),
                ("num", num_pipeline, numerical_features),
                ("pclass", pclass_pipeline, passenger_class_feature),
                ("cat", cat_pipeline, categorical_features),
                ("family", family_pipeline, family_features),
            ],
            remainder="drop",
        )

        # 4. Full Pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

        # 5. Fit the model
        # Logic check: Ensure X_train and y_train are not empty
        if X_train is None or y_train is None:
            raise ValueError("Training data is empty or not loaded correctly.")

        pipeline.fit(X_train, y_train)

        # 6. Save the model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        return {"success": True, "pipeline": pipeline, "error": None}

    except Exception as e:  # pylint: disable=broad-except
        # Capture any error (Data issues, File permissions, Math errors)
        return {"success": False, "pipeline": None, "error": str(e)}
