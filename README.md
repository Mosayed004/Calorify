# Calorify

Calorify is a multimodal deep learning project for food recognition and calorie estimation.

The system combines two main branches:

1. Image branch: predicts the food class from an uploaded image.
2. Text branch: estimates calories from the user's food description.

The final Django demo allows the user to upload a food image, write a meal description, and receive the predicted food class, confidence score, top-3 predictions, estimated calories, estimated calorie range, and warnings when needed.

---

## Project Objective

The goal of Calorify is to build an end-to-end deep learning pipeline that recognizes food images and estimates calories in an explainable way.

The project avoids direct exact image-to-calorie prediction because a single food image cannot reliably provide exact portion weight, hidden ingredients, sauces, oils, or preparation method.

Instead, Calorify uses:

- Image classification for food recognition.
- Text-based calorie estimation for ingredient and portion context.
- Integration logic to combine both outputs.
- Warning logic to detect possible image-text mismatch.

---

## Food Classes

The image model supports 10 food classes:

- burger
- pizza
- fries
- pasta
- rice
- chicken
- salad
- sandwich
- sushi
- steak

---

## Main Features

- Food image upload through Django.
- Food description text input.
- EfficientNetB0 image classification model.
- TF-IDF + Ridge calorie estimation model.
- Top-3 visual predictions.
- Confidence score.
- Estimated calorie value.
- Estimated calorie range.
- Image-text mismatch warning.
- Models comparison page.
- About page explaining the system and limitations.
- Final testing results saved in CSV format.
- Rubric alignment table saved in CSV format.

---

## Project Structure

```text
Calorify-Web/
│
├── .venv/
│
├── calorify_web/
│   ├── manage.py
│   │
│   ├── calorify_project/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   │
│   ├── predictor/
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── templates/
│   │   │   └── predictor/
│   │   │       ├── base.html
│   │   │       ├── home.html
│   │   │       ├── predict.html
│   │   │       ├── result.html
│   │   │       ├── models_comparison.html
│   │   │       └── about.html
│   │   │
│   │   └── static/
│   │       └── predictor/
│   │           └── style.css
│   │
│   ├── ml/
│   │   ├── model_loader.py
│   │   ├── image_prediction.py
│   │   ├── text_prediction.py
│   │   ├── consistency_check.py
│   │   └── final_pipeline.py
│   │
│   ├── media/
│   │   └── uploads/
│   │
│   └── results/
│       ├── final_testing_results.csv
│       ├── rubric_alignment_table.csv
│       └── phase5_final_checklist.csv
│
├── phase1_clean/
│   ├── models/
│   ├── metadata/
│   └── results/
│
├── phase2_text_calorie/
│   ├── models/
│   ├── data/
│   └── results/
│
├── phase3_multimodal_integration/
│   └── results/
│
├── requirements.txt
└── README.md

