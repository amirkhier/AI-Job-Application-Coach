"""
Base Celery task class with automatic retry and structured logging.

All application tasks should inherit from ``BaseTaskWithRetry`` to get:
  - Automatic retries (3× with exponential back-off + jitter)
  - Structured logging on success / failure
"""

from __future__ import annotations

import logging

from celery import Task

from app.celery_worker import celery_app

logger = logging.getLogger(__name__)


class BaseTaskWithRetry(Task):
    """Abstract base task with retry defaults and lifecycle hooks."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 120
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Task %s [%s] failed: %s", self.name, task_id, exc, exc_info=einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s [%s] completed successfully", self.name, task_id)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning("Task %s [%s] retrying due to: %s", self.name, task_id, exc)


# ── Trivial ping task for smoke-testing ──────────────────────────────────

@celery_app.task(name="tasks.ping")
def ping() -> str:
    """Health-check task — returns 'pong' if the worker is alive."""
    return "pong"
