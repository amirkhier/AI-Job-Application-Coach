"""
Tests for Phase 4 Step 3 — Application Tracker CRUD.

Covers:
  - Full CRUD lifecycle (create, read, update, delete)
  - Status workflow validation (valid/invalid transitions)
  - Follow-up reminder endpoint
  - Database methods (get_application_by_id, update_application, delete_application)
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
from fastapi.testclient import TestClient


def _get_app():
    from app.main import app
    return app


def _client():
    return TestClient(_get_app())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_db():
    """Create a mock DatabaseManager with sensible defaults."""
    db = MagicMock()
    db.ensure_connection.return_value = None
    db.execute_query.return_value = [{"test": 1}]
    return db


def _sample_application(id=1, status="applied"):
    """Return a realistic application dict as the DB would return it."""
    return {
        "id": id,
        "user_id": 1,
        "company_name": "Acme Corp",
        "position_title": "Backend Engineer",
        "job_url": "https://acme.com/jobs/123",
        "status": status,
        "application_date": date.today(),
        "follow_up_date": None,
        "notes": "Applied via website",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


# ===========================================================================
# Status Workflow Validation
# ===========================================================================

class TestApplicationStatusFlow:
    """Ensure status transitions follow the defined workflow."""

    def test_valid_transitions_from_applied(self):
        from app.main import APPLICATION_STATUS_FLOW
        assert "interviewing" in APPLICATION_STATUS_FLOW["applied"]
        assert "rejected" in APPLICATION_STATUS_FLOW["applied"]
        assert "withdrawn" in APPLICATION_STATUS_FLOW["applied"]

    def test_valid_transitions_from_interviewing(self):
        from app.main import APPLICATION_STATUS_FLOW
        assert "offer" in APPLICATION_STATUS_FLOW["interviewing"]
        assert "rejected" in APPLICATION_STATUS_FLOW["interviewing"]

    def test_terminal_states_have_no_transitions(self):
        from app.main import APPLICATION_STATUS_FLOW
        assert APPLICATION_STATUS_FLOW["rejected"] == []
        assert APPLICATION_STATUS_FLOW["withdrawn"] == []

    def test_cannot_go_backwards(self):
        from app.main import APPLICATION_STATUS_FLOW
        assert "applied" not in APPLICATION_STATUS_FLOW["interviewing"]


# ===========================================================================
# Database Methods
# ===========================================================================

class TestDatabaseApplicationMethods:
    """Test new DatabaseManager methods for application CRUD."""

    def test_get_application_by_id_exists(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        sample = _sample_application()
        db.execute_query = MagicMock(return_value=[sample])
        result = db.get_application_by_id(1)
        assert result is not None
        assert result["id"] == 1

    def test_get_application_by_id_not_found(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        db.execute_query = MagicMock(return_value=[])
        result = db.get_application_by_id(999)
        assert result is None

    def test_update_application_with_status(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        db.execute_update = MagicMock(return_value=1)
        db.ensure_connection = MagicMock()
        result = db.update_application(1, status="interviewing")
        assert result is True
        db.execute_update.assert_called_once()

    def test_update_application_with_multiple_fields(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        db.execute_update = MagicMock(return_value=1)
        db.ensure_connection = MagicMock()
        result = db.update_application(
            1, status="interviewing", notes="Phone screen scheduled",
            follow_up_date=date.today()
        )
        assert result is True

    def test_update_application_ignores_unknown_fields(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        db.execute_update = MagicMock(return_value=1)
        db.ensure_connection = MagicMock()
        # 'bogus_field' is not in the allowed set — should be silently ignored
        result = db.update_application(1, bogus_field="hacked")
        assert result is False  # no valid fields → no update

    def test_delete_application(self):
        from app.tools.database import DatabaseManager
        db = DatabaseManager()
        db.execute_update = MagicMock(return_value=1)
        db.ensure_connection = MagicMock()
        result = db.delete_application(1)
        assert result is True


# ===========================================================================
# PUT /applications/{id} Endpoint
# ===========================================================================

class TestUpdateApplicationEndpoint:
    """Test the fixed PUT endpoint."""

    def test_valid_status_transition(self):
        from app.main import app, get_database
        db = _mock_db()
        app_data = _sample_application(status="applied")
        updated_data = {**app_data, "status": "interviewing"}
        db.get_application_by_id = MagicMock(side_effect=[app_data, updated_data])
        db.update_application = MagicMock(return_value=True)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.put(
                "/applications/1",
                json={"status": "interviewing"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "interviewing"
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_invalid_status_transition_returns_422(self):
        from app.main import app, get_database
        db = _mock_db()
        db.get_application_by_id = MagicMock(return_value=_sample_application(status="applied"))
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.put(
                "/applications/1",
                json={"status": "offer"},  # can't jump applied → offer
            )
            assert resp.status_code == 422
            assert "Cannot transition" in resp.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_update_nonexistent_application_returns_404(self):
        from app.main import app, get_database
        db = _mock_db()
        db.get_application_by_id = MagicMock(return_value=None)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.put(
                "/applications/999",
                json={"status": "interviewing"},
            )
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_update_notes_only(self):
        from app.main import app, get_database
        db = _mock_db()
        app_data = _sample_application()
        updated_data = {**app_data, "notes": "Had a great chat"}
        db.get_application_by_id = MagicMock(side_effect=[app_data, updated_data])
        db.update_application = MagicMock(return_value=True)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.put(
                "/applications/1",
                json={"notes": "Had a great chat"},
            )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.pop(get_database, None)


# ===========================================================================
# DELETE /applications/{id} Endpoint
# ===========================================================================

class TestDeleteApplicationEndpoint:
    """Test the DELETE endpoint."""

    def test_delete_existing_application(self):
        from app.main import app, get_database
        db = _mock_db()
        db.get_application_by_id = MagicMock(return_value=_sample_application())
        db.delete_application = MagicMock(return_value=True)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.delete("/applications/1")
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_delete_nonexistent_application_returns_404(self):
        from app.main import app, get_database
        db = _mock_db()
        db.get_application_by_id = MagicMock(return_value=None)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.delete("/applications/999")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides.pop(get_database, None)


# ===========================================================================
# GET /applications/follow-ups Endpoint
# ===========================================================================

class TestFollowUpsEndpoint:
    """Test the follow-up reminder endpoint."""

    def test_returns_overdue_follow_ups(self):
        from app.main import app, get_database
        db = _mock_db()
        yesterday = date.today() - timedelta(days=1)
        apps = [
            {**_sample_application(id=1), "follow_up_date": yesterday, "status": "applied"},
            {**_sample_application(id=2), "follow_up_date": None, "status": "applied"},
        ]
        db.get_applications = MagicMock(return_value=apps)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.get("/applications/follow-ups?user_id=1")
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1
            assert data["follow_ups"][0]["id"] == 1
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_excludes_terminal_statuses(self):
        from app.main import app, get_database
        db = _mock_db()
        yesterday = date.today() - timedelta(days=1)
        apps = [
            {**_sample_application(id=1), "follow_up_date": yesterday, "status": "rejected"},
            {**_sample_application(id=2), "follow_up_date": yesterday, "status": "withdrawn"},
        ]
        db.get_applications = MagicMock(return_value=apps)
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.get("/applications/follow-ups?user_id=1")
            assert resp.status_code == 200
            assert resp.json()["count"] == 0
        finally:
            app.dependency_overrides.pop(get_database, None)

    def test_no_follow_ups_returns_empty(self):
        from app.main import app, get_database
        db = _mock_db()
        db.get_applications = MagicMock(return_value=[])
        app.dependency_overrides[get_database] = lambda: db
        try:
            client = TestClient(app)
            resp = client.get("/applications/follow-ups?user_id=1")
            assert resp.status_code == 200
            assert resp.json()["count"] == 0
        finally:
            app.dependency_overrides.pop(get_database, None)
