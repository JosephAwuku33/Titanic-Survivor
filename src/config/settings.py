"""
Configuration file for environment variable related stuff
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# load environment variables
load_dotenv()

DATASET_PATH_FROM_ENV = os.getenv("DATASET_PATH")
MODEL_PATH_FROM_ENV = os.getenv("MODEL_PATH")

if not DATASET_PATH_FROM_ENV:
    raise ValueError("DATASET_PATH environment variable not set or recognized")

if not MODEL_PATH_FROM_ENV:
    raise ValueError("MODEL_PATH environment variable not set or recognized")


DATASET_PATH = Path(DATASET_PATH_FROM_ENV)
MODEL_PATH = Path(MODEL_PATH_FROM_ENV)
