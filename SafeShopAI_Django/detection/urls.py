from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("result/<str:filename>/", views.result, name="result"),
]
