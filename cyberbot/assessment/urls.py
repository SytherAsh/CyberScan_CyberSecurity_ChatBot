from django.urls import path
from . import views

urlpatterns = [
    path('', views.query_view, name='query'),  # Root URL for form
    path('query/', views.query_view, name='query_api'),  # Optional: keep API endpoint
    # path('upload/', views.upload_pdf, name='upload_pdf'),
]