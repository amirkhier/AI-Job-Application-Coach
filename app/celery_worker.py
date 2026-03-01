"""
Celery application instance for the AI Job Application Coach.

This module creates the ``celery_app`` singleton that workers and the
FastAPI app both import.  The broker URL, result backend, and tuning
knobs all flow from ``app.config.settings``.

Start a worker locally::

    celery -A app.celery_worker worker --loglevel=info

Or via Docker Compose::

    docker compose up celery_worker
"""

from __future__ import annotations

from celery import Celery

from app.config import settings

celery_app = Celery(
    "job_coach",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                  # re-deliver if worker crashes
    worker_prefetch_multiplier=1,         # fair scheduling
    result_expires=86400,                 # 24 h TTL on results
    task_soft_time_limit=300,             # 5 min soft limit
    task_time_limit=600,                  # 10 min hard kill
)

# Auto-discover task modules inside app/tasks/
celery_app.autodiscover_tasks(["app.tasks"])
