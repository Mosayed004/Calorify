from pathlib import Path

import numpy as np
from PIL import Image

from .model_loader import load_class_names, load_image_model


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_food(image_path, top_k=3):
    model = load_image_model()
    class_names = load_class_names()

    image_array = preprocess_image(image_path)
    probabilities = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = []
    for index in top_indices:
        top_predictions.append(
            {
                "class_name": class_names[index],
                "confidence": float(probabilities[index]),
                "confidence_percent": round(float(probabilities[index]) * 100, 2),
            }
        )

    best_prediction = top_predictions[0]

    return {
        "predicted_food": best_prediction["class_name"],
        "confidence": best_prediction["confidence"],
        "confidence_percent": best_prediction["confidence_percent"],
        "top_predictions": top_predictions,
    }