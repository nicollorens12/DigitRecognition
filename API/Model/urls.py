from django.urls import path
from .views import predict_digit

urlpatterns = [
    path('predict/', predict_digit, name='predict_digit'),
]
