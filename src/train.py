"""
Module for handling preprocessing of data and training of the model.

This script loads the Titanic dataset, creates preprocessing pipelines for
different feature groups, combines them into a single preprocessing pipeline,
and trains a LogisticRegression model on the processed data.
"""

from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.load import load_and_split_data
from src.utils.transformer import FamilyFeatures

# Load training and testing data
BASE_DIR = Path("data/raw") / "titanic.csv"

X_train, X_test, y_train, y_test = load_and_split_data(BASE_DIR)

# Feature groups
age_feature = ["Age"]
numerical_features = ["Fare"]
categorical_features = ["Embarked", "Sex"]
passenger_class_feature = ["Pclass"]
family_features = ["SibSp", "Parch"]

# Pipelines for each feature group
age_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

num_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

pclass_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ]
)

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

# Combine all feature pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("age", age_pipeline, age_feature),
        ("num", num_pipeline, numerical_features),
        ("pclass", pclass_pipeline, passenger_class_feature),
        ("cat", cat_pipeline, categorical_features),
        ("family", family_pipeline, family_features),
    ],
    remainder="drop",  # Drop any columns not specified in transformers
)

# Create full pipeline with preprocessing and model
pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Fit the pipeline
pipeline.fit(X_train, y_train)
print(f"Pipeline:\n{pipeline}\n")

# Display model coefficients
print("Model Coefficients:")
print(f"{pipeline.named_steps['model'].coef_}\n")

# Create models directory and save the pipeline
Path("models").mkdir(exist_ok=True, parents=True)
joblib.dump(pipeline, "models/titanic_pipeline.pkl")
print("Successfully trained and saved the pipeline to 'models/titanic_pipeline.pkl'")
