"""
Tests for Phase 4 Step 4 — Celery Infrastructure.

Covers:
  - Celery app configuration
  - Base task class attributes
  - Ping task execution (eager mode)
  - Task status endpoint
"""

import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# Celery App Configuration
# ===========================================================================

class TestCeleryAppConfig:
    """Validate the celery_app singleton configuration."""

    def test_celery_app_imports(self):
        from app.celery_worker import celery_app
        assert celery_app is not None
        assert celery_app.main == "job_coach"

    def test_celery_serialiser_is_json(self):
        from app.celery_worker import celery_app
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"

    def test_celery_tracks_started(self):
        from app.celery_worker import celery_app
        assert celery_app.conf.task_track_started is True

    def test_celery_acks_late(self):
        from app.celery_worker import celery_app
        assert celery_app.conf.task_acks_late is True

    def test_celery_time_limits(self):
        from app.celery_worker import celery_app
        assert celery_app.conf.task_soft_time_limit == 300
        assert celery_app.conf.task_time_limit == 600

    def test_celery_result_expires(self):
        from app.celery_worker import celery_app
        assert celery_app.conf.result_expires == 86400


# ===========================================================================
# Base Task
# ===========================================================================

class TestBaseTask:
    """Validate BaseTaskWithRetry attributes."""

    def test_base_task_retry_config(self):
        from app.tasks.base import BaseTaskWithRetry
        assert BaseTaskWithRetry.autoretry_for == (Exception,)
        assert BaseTaskWithRetry.retry_kwargs == {"max_retries": 3}
        assert BaseTaskWithRetry.retry_backoff is True

    def test_ping_task_eager(self):
        """Run the ping task in eager mode (synchronous, no broker)."""
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
        try:
            from app.tasks.base import ping
            result = ping.delay()
            assert result.get() == "pong"
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ===========================================================================
# Task Status Endpoint
# ===========================================================================

class TestTaskStatusEndpoint:
    """Test /tasks/{task_id}/status endpoint."""

    @patch("app.main.AsyncResult" if False else "celery.result.AsyncResult")
    def test_task_status_success(self, mock_async_result):
        """Completed task should return SUCCESS with result data."""
        from fastapi.testclient import TestClient
        from app.main import app

        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.successful.return_value = True
        mock_result.failed.return_value = False
        mock_result.result = {"analysis": "done"}
        mock_async_result.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/tasks/test-task-123/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "SUCCESS"
        assert data["result"] == {"analysis": "done"}

    @patch("celery.result.AsyncResult")
    def test_task_status_pending(self, mock_async_result):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.successful.return_value = False
        mock_result.failed.return_value = False
        mock_async_result.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/tasks/test-task-456/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"

    @patch("celery.result.AsyncResult")
    def test_task_status_failed(self, mock_async_result):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.successful.return_value = False
        mock_result.failed.return_value = True
        mock_result.result = RuntimeError("LLM timeout")
        mock_async_result.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/tasks/test-task-789/status")
        data = resp.json()
        assert data["status"] == "FAILURE"
        assert "LLM timeout" in data["error"]

    @patch("celery.result.AsyncResult")
    def test_task_status_progress(self, mock_async_result):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_result = MagicMock()
        mock_result.status = "PROGRESS"
        mock_result.successful.return_value = False
        mock_result.failed.return_value = False
        mock_result.info = {"step": 2, "total": 5, "detail": "Improvement suggestions"}
        mock_async_result.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/tasks/test-task-progress/status")
        data = resp.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["step"] == 2

    def test_legacy_result_endpoint(self):
        """The /result/{id} endpoint should still work (alias)."""
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get("/result/some-id")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "some-id"
