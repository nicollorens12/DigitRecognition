from django.urls import path
from Model.views import predict_digit

urlpatterns = [
    path('predict/', predict_digit, name='predict_digit'),
]
