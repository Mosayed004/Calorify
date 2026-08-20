FOOD_KEYWORDS = {
    "burger": ["burger", "cheeseburger", "beef patty", "bun", "lettuce wrap", "mayo", "ketchup"],
    "pizza": ["pizza", "pepperoni", "mozzarella", "crust", "tomato sauce"],
    "fries": ["fries", "french fries", "potato fries", "chips"],
    "pasta": ["pasta", "spaghetti", "penne", "bolognese", "pesto", "noodles"],
    "rice": ["rice", "white rice", "fried rice", "steamed rice"],
    "chicken": ["chicken", "grilled chicken", "chicken breast", "fried chicken"],
    "salad": ["salad", "lettuce", "cucumber", "tomato", "dressing"],
    "sandwich": ["sandwich", "toast", "bread", "sub", "wrap"],
    "sushi": ["sushi", "maki", "nori", "salmon roll", "soy sauce"],
    "steak": ["steak", "beef steak", "grilled steak", "sirloin"],
}


def check_empty_text(user_text):
    if not user_text or not str(user_text).strip():
        return "Food description is empty, so calorie estimation may be unavailable."

    return None


def check_low_confidence(confidence):
    if confidence < 0.60:
        return "Image classification confidence is relatively low. Please verify the predicted food type."

    return None


def check_calorie_plausibility(calories):
    if calories is None:
        return "Estimated calories could not be calculated from the provided text."

    if calories < 80:
        return "Estimated calories are very low. Please check if the meal description is complete."

    if calories > 1800:
        return "Estimated calories are very high. Please check portion size and ingredients."

    return None


def check_text_image_consistency(predicted_food, user_text):
    if not predicted_food or not user_text:
        return None

    text = str(user_text).lower()
    predicted_food = str(predicted_food).lower()

    predicted_keywords = FOOD_KEYWORDS.get(predicted_food, [])

    if any(keyword in text for keyword in predicted_keywords):
        return None

    other_food_mentions = []

    for food_class, keywords in FOOD_KEYWORDS.items():
        if food_class == predicted_food:
            continue

        if any(keyword in text for keyword in keywords):
            other_food_mentions.append(food_class)

    if other_food_mentions:
        mentioned = ", ".join(sorted(set(other_food_mentions)))
        return (
            f"The image model predicted '{predicted_food}', but the text description "
            f"appears to mention: {mentioned}. Please check if the image and description match."
        )

    return None


def collect_warnings(predicted_food, confidence, user_text, calories):
    warning_checks = [
        check_empty_text(user_text),
        check_low_confidence(confidence),
        check_calorie_plausibility(calories),
        check_text_image_consistency(predicted_food, user_text),
    ]

    return [warning for warning in warning_checks if warning]