from django.urls import path, include

urlpatterns = [
    path('api/assistant/', include('chat.urls')),
]
