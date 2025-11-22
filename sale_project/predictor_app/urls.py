from django.urls import path
from . import views

app_name = "predictor_app"

urlpatterns = [
    path("", views.upload_and_predict, name="upload_and_predict"),
]
