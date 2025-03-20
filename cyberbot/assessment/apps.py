from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class AssessmentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'assessment'

    # def ready(self):
    #     """Initialize machine learning models on app startup."""
    #     try:
    #         from .ml_utils import initialize_models
    #         initialize_models()
    #         logger.info("Models initialized successfully.")
    #     except Exception as e:
    #         logger.error(f"Failed to initialize models: {e}")
    #         raise  # Re-raise to halt startup if critical