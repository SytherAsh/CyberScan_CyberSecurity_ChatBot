from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('query/', views.query_view, name='query_api'),  # Optional: keep API endpoint
]