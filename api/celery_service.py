from .celery_app import celery_app
# Import tasks here to ensure they are registered when worker starts
from . import tasks 
