import re

from .model_loader import load_text_model


MIN_CALORIES = 50
MAX_CALORIES = 2500


def preprocess_text(user_text):
    if user_text is None:
        return ""

    text = str(user_text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_estimated_range(calories):
    lower = max(MIN_CALORIES, round(calories * 0.9))
    upper = min(MAX_CALORIES, round(calories * 1.1))

    return {
        "lower": lower,
        "upper": upper,
        "label": f"{lower}–{upper} kcal",
    }


def predict_calories(user_text):
    cleaned_text = preprocess_text(user_text)

    if not cleaned_text:
        return {
            "estimated_calories": None,
            "estimated_range": None,
            "cleaned_text": cleaned_text,
        }

    model = load_text_model()
    prediction = model.predict([cleaned_text])[0]

    calories = round(float(prediction), 2)
    calories = max(MIN_CALORIES, min(MAX_CALORIES, calories))

    return {
        "estimated_calories": calories,
        "estimated_range": make_estimated_range(calories),
        "cleaned_text": cleaned_text,
    }
