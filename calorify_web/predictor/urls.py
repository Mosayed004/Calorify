from django.urls import path
from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.home_view, name="home"),
    path("predict/", views.predict_view, name="predict"),
    path("models-comparison/", views.models_comparison_view, name="models_comparison"),
    path("about/", views.about_view, name="about"),
]