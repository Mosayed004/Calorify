from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from ml.final_pipeline import final_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def home_view(request):
    return render(request, "predictor/home.html")


def predict_view(request):
    if request.method == "POST":
        food_image = request.FILES.get("food_image")
        food_description = request.POST.get("food_description", "").strip()

        errors = []

        if not food_image:
            errors.append("Please upload a food image.")

        if not food_description:
            errors.append("Please enter a food description.")

        if errors:
            return render(
                request,
                "predictor/predict.html",
                {
                    "errors": errors,
                    "food_description": food_description,
                },
            )

        upload_dir = settings.MEDIA_ROOT / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        storage = FileSystemStorage(
            location=upload_dir,
            base_url=settings.MEDIA_URL + "uploads/",
        )

        saved_filename = storage.save(food_image.name, food_image)
        uploaded_image_url = storage.url(saved_filename)
        uploaded_image_path = upload_dir / saved_filename

        prediction_result = final_pipeline(uploaded_image_path, food_description)

        return render(
            request,
            "predictor/result.html",
            {
                "uploaded_image_url": uploaded_image_url,
                "food_description": food_description,
                "prediction": prediction_result,
            },
        )

    return render(request, "predictor/predict.html")


def _load_csv_records(file_path, selected_columns):
    if not file_path.exists():
        return []

    dataframe = pd.read_csv(file_path)

    available_columns = [
        column for column in selected_columns if column in dataframe.columns
    ]

    if not available_columns:
        return []

    dataframe = dataframe[available_columns]
    dataframe = dataframe.fillna("")

    records = []

    for _, row in dataframe.iterrows():
        record = {}

        for column in available_columns:
            value = row[column]

            if hasattr(value, "item"):
                value = value.item()

            record[column] = value

        records.append(record)

    return records

def models_comparison_view(request):
    phase1_comparison_path = (
        PROJECT_ROOT
        / "phase1_clean"
        / "results"
        / "model_comparison"
        / "phase1_model_comparison.csv"
    )

    phase1_best_path = (
        PROJECT_ROOT
        / "phase1_clean"
        / "results"
        / "model_comparison"
        / "phase1_best_model_summary.csv"
    )

    phase2_comparison_path = (
        PROJECT_ROOT
        / "phase2_text_calorie"
        / "results"
        / "phase2_validation_model_comparison.csv"
    )

    phase2_best_path = (
        PROJECT_ROOT
        / "phase2_text_calorie"
        / "results"
        / "phase2_best_model_selection.csv"
    )

    image_models = _load_csv_records(
        phase1_comparison_path,
        [
            "rank",
            "model_name",
            "model_type",
            "test_accuracy_percent",
            "macro_f1_percent",
            "weighted_f1_percent",
            "correct_predictions",
            "wrong_predictions",
        ],
    )

    best_image_model_rows = _load_csv_records(
        phase1_best_path,
        [
            "best_visual_model",
            "model_type",
            "test_accuracy_percent",
            "macro_f1_percent",
            "weighted_f1_percent",
            "final_decision",
        ],
    )

    text_models = _load_csv_records(
        phase2_comparison_path,
        [
            "model_name",
            "model_type",
            "uses_text",
            "deep_learning",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "notes",
        ],
    )

    best_text_model_rows = _load_csv_records(
        phase2_best_path,
        [
            "selected_model_name",
            "selected_model_type",
            "selection_metric",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "selection_reason",
        ],
    )

    context = {
        "image_models": image_models,
        "best_image_model": best_image_model_rows[0] if best_image_model_rows else None,
        "text_models": text_models,
        "best_text_model": best_text_model_rows[0] if best_text_model_rows else None,
    }

    return render(request, "predictor/models_comparison.html", context)


def about_view(request):
    return render(request, "predictor/about.html")