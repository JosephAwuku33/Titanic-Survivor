# Titanic Survivor Prediction
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/JosephAwuku33/Titanic-Survivor)

## Overview

This repository contains a machine learning project to predict the survival of passengers on the RMS Titanic. The project encompasses the entire ML lifecycle, from data exploration and feature engineering to model training and deployment via a RESTful API.

The core of this project is a Scikit-learn pipeline that preprocesses the data and trains a Logistic Regression model. The trained model is then served using a FastAPI application, allowing for real-time survival predictions based on passenger attributes.

## Features

- **Exploratory Data Analysis (EDA):** A Jupyter notebook (`notebooks/exploration.ipynb`) provides detailed analysis and visualizations of the dataset.
- **Scikit-learn Pipeline:** A robust pipeline handles missing value imputation, scaling of numerical features, and one-hot encoding of categorical features.
- **Custom Feature Engineering:** Includes a custom `FamilyFeatures` transformer to create new features like `family_size` and `isAlone`.
- **Model Training & Evaluation:** Scripts to train the model (`src/train.py`) and evaluate its performance (`src/evaluate.py`).
- **RESTful API:** A FastAPI application (`app/`) serves the trained model, providing a simple endpoint for predictions.
- **Data Validation:** Pydantic models ensure that the data sent to the API is valid.

## Repository Structure

```
.
├── app/                  # FastAPI application code
│   ├── router/           # API router for predictions
│   └── schema/           # Pydantic data models for validation
├── data/raw/             # Directory for the raw dataset (titanic.csv)
├── models/               # Saved model pipeline (titanic_pipeline.pkl)
├── notebooks/            # Jupyter notebooks for EDA
│   └── exploration.ipynb
├── src/                  # Core ML pipeline source code
│   ├── utils/            # Utility functions and custom transformers
│   ├── train.py          # Script to train the model
│   ├── evaluate.py       # Script to evaluate the model
│   └── predict.py        # Script for making predictions
└── requirements.txt      # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/josephawuku33/titanic-survivor.git
    cd titanic-survivor
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Add the dataset:**
    Create a `data/raw` directory and place the `titanic.csv` dataset inside it. The project expects the file at `data/raw/titanic.csv`.

## Usage

### 1. Train the Model

Run the training script from the root directory. This will process the data, train the pipeline, and save the final model artifact to `models/titanic_pipeline.pkl`.

```sh
python -m src.train
```

### 2. Evaluate the Model

To see the performance of the trained model on the test set, run the evaluation script:

```sh
python -m src.evaluate
```

This will print a classification report to the console.

### 3. Run the API

Start the FastAPI server using Uvicorn:

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will now be accessible at `http://127.0.0.1:8000`.

## API Endpoint

### Predict Survival

Make a `POST` request to the `/predict` endpoint with the passenger's data to get a survival prediction.

-   **Endpoint:** `/predict`
-   **Method:** `POST`
-   **Request Body:**

    The request body must be a JSON object containing the following fields:

    -   `Age` (integer)
    -   `Fare` (float)
    -   `Sex` (string: `"male"` or `"female"`)
    -   `Pclass` (integer: `1`, `2`, or `3`)
    -   `Embarked` (string: `"Q"`, `"C"`, or `"S"`)
    -   `SibSp` (integer)
    -   `Parch` (integer)

-   **Example `curl` Request:**

    ```sh
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "Age": 30,
        "Fare": 75.5,
        "Sex": "female",
        "Pclass": 1,
        "Embarked": "C",
        "SibSp": 1,
        "Parch": 0
      }'
    ```

-   **Success Response:**

    ```json
    {
      "Did the person most likely survive": "yes"
    }