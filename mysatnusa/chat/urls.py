from django.urls import path
from .views import gemini_prompt_view

urlpatterns = [
    path('prompt_view', gemini_prompt_view, name='gemini_prompt_view'),
]
