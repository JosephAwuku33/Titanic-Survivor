"""
Configuration file for environment variable related stuff
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# load environment variables
load_dotenv()

PY_ENV = os.getenv("PY_ENV")
LOCAL_DATASET_PATH_FROM_ENV = os.getenv("LOCAL_DATASET_PATH")
HF_DATASET_PATH_FROM_ENV = os.getenv("HF_DATASET_PATH")
MODEL_PATH_FROM_ENV = os.getenv("MODEL_PATH")

if not PY_ENV:
    raise ValueError("Python Environment variable is not set or recognized")

if not LOCAL_DATASET_PATH_FROM_ENV or not HF_DATASET_PATH_FROM_ENV:
    raise ValueError("Dataset path for the environment isn't set or recognized")

if not MODEL_PATH_FROM_ENV:
    raise ValueError("MODEL_PATH environment variable not set or recognized")


if PY_ENV == "production":
    DATASET_PATH = Path(str(HF_DATASET_PATH_FROM_ENV))
else:
    DATASET_PATH = Path(str(LOCAL_DATASET_PATH_FROM_ENV))

MODEL_PATH = Path(str(MODEL_PATH_FROM_ENV))
