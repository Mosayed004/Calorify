from .consistency_check import collect_warnings
from .image_prediction import predict_food
from .text_prediction import predict_calories


def final_pipeline(image_path, user_text):
    image_result = predict_food(image_path)
    calorie_result = predict_calories(user_text)

    warnings = collect_warnings(
        predicted_food=image_result["predicted_food"],
        confidence=image_result["confidence"],
        user_text=user_text,
        calories=calorie_result["estimated_calories"],
    )

    return {
        "predicted_food": image_result["predicted_food"],
        "confidence": image_result["confidence"],
        "confidence_percent": image_result["confidence_percent"],
        "top_predictions": image_result["top_predictions"],
        "estimated_calories": calorie_result["estimated_calories"],
        "estimated_range": calorie_result["estimated_range"],
        "cleaned_text": calorie_result["cleaned_text"],
        "warnings": warnings,
    }