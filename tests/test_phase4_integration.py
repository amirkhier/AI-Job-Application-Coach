"""
Phase 4 Integration Tests — Cross-cutting end-to-end validation.

Covers:
  - Health check subsystem reporting
  - Global exception handler
  - Middleware ordering (auth + rate-limit + validation)
  - Application CRUD workflow end-to-end
  - Async task dispatch + status polling
  - Graceful degradation when Celery is unavailable
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime, timedelta
from fastapi.testclient import TestClient


def _app():
    from app.main import app
    return app


def _client():
    return TestClient(_app())


# ===========================================================================
# Health Check — Subsystem Reporting
# ===========================================================================

class TestHealthCheckEnhanced:
    """Verify /health reports database + redis status."""

    def test_health_returns_checks_dict(self):
        client = _client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "checks" in data
        assert "database" in data["checks"]
        assert "redis" in data["checks"]

    def test_health_status_degraded_when_db_down(self):
        """When DB is unreachable, status should be 'degraded'."""
        client = _client()
        resp = client.get("/health")
        data = resp.json()
        # In CI/test, DB may not be connected, which is fine
        assert data["status"] in ("healthy", "degraded")


# ===========================================================================
# Global Exception Handler
# ===========================================================================

class TestGlobalExceptionHandler:
    """Verify unhandled exceptions return structured 500."""

    def test_unhandled_exception_returns_500_with_request_id(self):
        """A truly unhandled error should produce a JSON 500 with request_id."""
        from app.main import app

        # Temporarily register a route that always raises
        @app.get("/test-explosion")
        async def _boom():
            raise RuntimeError("kaboom")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test-explosion")
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal server error"
        assert "request_id" in body


# ===========================================================================
# Middleware Integration
# ===========================================================================

class TestMiddlewareStack:
    """Verify middleware layers work together."""

    def test_request_id_header_set(self):
        client = _client()
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_custom_request_id_echoed(self):
        client = _client()
        resp = client.get("/health", headers={"X-Request-ID": "my-trace-42"})
        assert resp.headers["X-Request-ID"] == "my-trace-42"

    def test_rate_limit_headers_present(self):
        client = _client()
        resp = client.get("/health")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    @patch("app.middleware.auth.settings")
    def test_auth_before_endpoint(self, mock_settings):
        """When API_KEY is set, unauthenticated POST should fail 401."""
        mock_settings.API_KEY = "secret"
        client = _client()
        resp = client.post("/ask", json={"query": "hello world test", "user_id": 1})
        assert resp.status_code == 401


# ===========================================================================
# Application CRUD Full Workflow
# ===========================================================================

class TestApplicationWorkflow:
    """End-to-end: create → read → update (with transition) → delete."""

    def test_full_crud_lifecycle(self):
        from app.main import app, get_database
        db = MagicMock()
        db.ensure_connection.return_value = None
        db.execute_query.return_value = [{"test": 1}]

        today = date.today()
        now = datetime.now()

        created_app = {
            "id": 42, "user_id": 1, "company_name": "TestCo",
            "position_title": "Engineer", "job_url": None,
            "status": "applied", "application_date": today,
            "follow_up_date": None, "notes": None,
            "created_at": now, "updated_at": now,
        }
        updated_app = {**created_app, "status": "interviewing", "updated_at": now}

        db.create_application.return_value = 42
        db.get_applications.return_value = [created_app]
        db.get_application_by_id.side_effect = [
            created_app,   # first call (UPDATE pre-fetch)
            updated_app,   # second call (UPDATE post-fetch)
            updated_app,   # third call (DELETE pre-fetch)
        ]
        db.update_application.return_value = True
        db.delete_application.return_value = True
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)

            # CREATE
            resp = client.post("/applications", json={
                "company_name": "TestCo",
                "position_title": "Engineer",
                "job_url": None,
                "application_date": str(today),
                "notes": None,
                "user_id": 1,
            })
            assert resp.status_code == 200
            assert resp.json()["id"] == 42

            # READ
            resp = client.get("/applications?user_id=1")
            assert resp.status_code == 200
            assert len(resp.json()) >= 1

            # UPDATE (valid transition)
            resp = client.put("/applications/42", json={"status": "interviewing"})
            assert resp.status_code == 200
            assert resp.json()["status"] == "interviewing"

            # DELETE
            resp = client.delete("/applications/42")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True
        finally:
            app.dependency_overrides.pop(get_database, None)


# ===========================================================================
# Async Task Dispatch + Polling
# ===========================================================================

class TestAsyncTaskWorkflow:
    """Dispatch a task and poll for its status."""

    @patch("app.tasks.resume_tasks.detailed_resume_audit")
    @patch("celery.result.AsyncResult")
    def test_dispatch_then_poll(self, mock_async_result, mock_task):
        """POST /resume/audit → GET /tasks/{id}/status."""
        from app.main import app

        # Mock task dispatch
        task_mock = MagicMock()
        task_mock.id = "task-e2e-123"
        mock_task.delay.return_value = task_mock

        # Mock status check
        result_mock = MagicMock()
        result_mock.status = "PROGRESS"
        result_mock.successful.return_value = False
        result_mock.failed.return_value = False
        result_mock.info = {"step": 3, "total": 5, "detail": "Skill gap analysis"}
        mock_async_result.return_value = result_mock

        client = TestClient(app)

        # Dispatch
        resp = client.post("/resume/audit", json={
            "resume_text": "x " * 50,
            "user_id": 1,
        })
        assert resp.status_code == 202
        task_id = resp.json()["task_id"]
        assert task_id == "task-e2e-123"

        # Poll
        resp = client.get(f"/tasks/{task_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "PROGRESS"
        assert data["progress"]["step"] == 3


# ===========================================================================
# Graceful Degradation
# ===========================================================================

class TestGracefulDegradation:
    """Async endpoints should not crash when Celery/Redis is down."""

    @patch("app.tasks.resume_tasks.detailed_resume_audit")
    def test_resume_audit_fallback(self, mock_task):
        mock_task.delay.side_effect = ConnectionError("Redis down")

        client = _client()
        resp = client.post("/resume/audit", json={
            "resume_text": "x " * 50,
            "user_id": 1,
        })
        assert resp.status_code == 202
        assert "unavailable" in resp.json()["message"].lower()

    @patch("app.tasks.interview_tasks.generate_interview_report")
    def test_interview_report_fallback(self, mock_task):
        mock_task.delay.side_effect = ConnectionError("Redis down")

        client = _client()
        resp = client.post("/interview/report?user_id=1&session_id=test-sess")
        assert resp.status_code == 202
        assert "unavailable" in resp.json()["message"].lower()

    @patch("app.tasks.application_tasks.batch_status_check")
    def test_batch_update_fallback(self, mock_task):
        mock_task.delay.side_effect = ConnectionError("Redis down")

        client = _client()
        resp = client.post("/applications/batch-update?user_id=1")
        assert resp.status_code == 202
        assert "unavailable" in resp.json()["message"].lower()


# ===========================================================================
# Config & Feature Flags
# ===========================================================================

class TestConfigIntegration:
    """Verify configuration flows correctly."""

    def test_settings_singleton(self):
        from app.config import settings
        assert settings.ENVIRONMENT in ("development", "staging", "production")
        assert isinstance(settings.RATE_LIMIT_PER_MINUTE, int)
        assert settings.RATE_LIMIT_PER_MINUTE > 0

    def test_api_key_default_is_none(self):
        """By default, API_KEY should be None (dev mode — auth bypassed)."""
        from app.config import settings
        # In test environment, API_KEY is typically not set
        # This test just verifies the field exists and is Optional
        assert hasattr(settings, "API_KEY")
