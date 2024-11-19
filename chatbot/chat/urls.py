from django.urls import path
from .views import chatbot_api

urlpatterns = [
    path('', chatbot_api, name='chatbot_api'),
]
