import json
import os
from functools import lru_cache
from pathlib import Path

import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_MODEL_PATH = PROJECT_ROOT / "phase1_clean" / "models" / "efficientnetb0_transfer_learning.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "phase1_clean" / "models" / "best_model" / "class_names.json"
TEXT_MODEL_PATH = PROJECT_ROOT / "phase2_text_calorie" / "models" / "tfidf_ridge_best_calorie_model.joblib"


def get_model_paths():
    return {
        "image_model": IMAGE_MODEL_PATH,
        "class_names": CLASS_NAMES_PATH,
        "text_model": TEXT_MODEL_PATH,
    }


@lru_cache(maxsize=1)
def load_image_model():
    if not IMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Image model not found: {IMAGE_MODEL_PATH}")

    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


@lru_cache(maxsize=1)
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError("class_names.json must contain a list of class names.")

    return class_names


@lru_cache(maxsize=1)
def load_text_model():
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Text model not found: {TEXT_MODEL_PATH}")

    return joblib.load(TEXT_MODEL_PATH)