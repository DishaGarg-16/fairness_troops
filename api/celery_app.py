import os
from celery import Celery

# Get Redis Config from Environment
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Initialize Celery
celery_app = Celery(
    "fairness_troops",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Optional: Route specific tasks to queues if needed
    # task_routes={
    #     "api.tasks.*": {"queue": "fairness_queue"},
    # }
)

# Auto-discover tasks in packages
# We will create a tasks module next
celery_app.autodiscover_tasks(['api'])
