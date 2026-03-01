"""
Tests for Phase 4 Step 6 — Background Report Generation.

Covers:
  - Interview performance report task
  - Batch application status check task
  - Follow-up reminder generation task
  - Endpoint dispatching for /interview/report and /applications/batch-update
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime, timedelta


# ===========================================================================
# Interview Report Task
# ===========================================================================

class TestInterviewReportTask:
    """Test generate_interview_report Celery task."""

    @patch("app.tools.database.DatabaseManager")
    def test_report_task_returns_complete_report(self, MockDB):
        """Run in eager mode and verify report structure."""
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_db = MockDB.return_value
        mock_db.get_interview_session.return_value = {
            "session_id": "sess-1",
            "role": "Backend Engineer",
            "level": "mid",
            "questions": [
                {"id": "q1", "question": "Tell me about yourself"},
                {"id": "q2", "question": "Describe a challenge"},
            ],
            "answers": [
                {
                    "question_id": "q1",
                    "answer": "I am a developer...",
                    "evaluation": {"overall_score": 7.0, "strength_areas": ["clarity"], "improvement_areas": ["depth"]},
                },
                {
                    "question_id": "q2",
                    "answer": "I faced a difficult bug...",
                    "evaluation": {"overall_score": 6.0, "strength_areas": ["specificity"], "improvement_areas": ["STAR method"]},
                },
            ],
        }

        try:
            from app.tasks.interview_tasks import generate_interview_report

            result = generate_interview_report.delay(user_id=1, session_id="sess-1")
            report = result.get()
            assert report["report_complete"] is True
            assert report["overview"]["average_score"] == 6.5
            assert report["overview"]["total_questions"] == 2
            assert report["overview"]["total_answered"] == 2
            assert report["overview"]["performance_level"] == "good"
            assert len(report["per_question_breakdown"]) == 2
            assert "improvement_plan" in report
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)

    @patch("app.tools.database.DatabaseManager")
    def test_report_task_session_not_found(self, MockDB):
        """When session doesn't exist, return error."""
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_db = MockDB.return_value
        mock_db.get_interview_session.return_value = None

        try:
            from app.tasks.interview_tasks import generate_interview_report

            result = generate_interview_report.delay(user_id=1, session_id="nonexistent")
            report = result.get()
            assert report.get("audit_complete") is False or "error" in report
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ===========================================================================
# Batch Application Status Check Task
# ===========================================================================

class TestBatchStatusCheckTask:
    """Test batch_status_check Celery task."""

    @patch("app.tools.database.DatabaseManager")
    def test_detects_overdue_and_stale(self, MockDB):
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_db = MockDB.return_value
        yesterday = date.today() - timedelta(days=1)
        old_date = datetime.now() - timedelta(days=45)
        mock_db.get_applications.return_value = [
            {
                "id": 1, "company_name": "Acme", "position_title": "Dev",
                "status": "applied", "follow_up_date": yesterday,
                "updated_at": datetime.now(),
            },
            {
                "id": 2, "company_name": "Beta", "position_title": "SRE",
                "status": "interviewing", "follow_up_date": None,
                "updated_at": old_date,
            },
            {
                "id": 3, "company_name": "Gamma", "position_title": "PM",
                "status": "rejected", "follow_up_date": yesterday,
                "updated_at": datetime.now(),
            },
        ]

        try:
            from app.tasks.application_tasks import batch_status_check

            result = batch_status_check.delay(user_id=1)
            data = result.get()
            assert data["check_complete"] is True
            assert data["active_applications"] == 2
            assert len(data["overdue_follow_ups"]) == 1
            assert data["overdue_follow_ups"][0]["company"] == "Acme"
            assert len(data["stale_applications"]) == 1
            assert data["stale_applications"][0]["company"] == "Beta"
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ===========================================================================
# Follow-Up Reminder Task
# ===========================================================================

class TestFollowUpReminderTask:
    """Test generate_follow_up_reminders task."""

    @patch("app.tools.database.DatabaseManager")
    def test_generates_reminders_with_urgency(self, MockDB):
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_db = MockDB.return_value
        mock_db.get_applications.return_value = [
            {
                "id": 1, "company_name": "Acme", "position_title": "Dev",
                "status": "applied",
                "follow_up_date": date.today() - timedelta(days=10),
            },
            {
                "id": 2, "company_name": "Beta", "position_title": "SRE",
                "status": "applied",
                "follow_up_date": date.today() - timedelta(days=2),
            },
        ]

        try:
            from app.tasks.application_tasks import generate_follow_up_reminders

            result = generate_follow_up_reminders.delay(user_id=1)
            data = result.get()
            assert data["count"] == 2
            urgencies = [r["urgency"] for r in data["reminders"]]
            assert "high" in urgencies  # 10 days overdue
            assert "low" in urgencies   # 2 days overdue
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ===========================================================================
# Endpoints
# ===========================================================================

class TestReportEndpoints:
    """Test the dispatch endpoints."""

    @patch("app.tasks.interview_tasks.generate_interview_report")
    def test_interview_report_endpoint(self, mock_task):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_async = MagicMock()
        mock_async.id = "task-report-1"
        mock_task.delay.return_value = mock_async

        client = TestClient(app)
        resp = client.post("/interview/report?user_id=1&session_id=sess-1")
        assert resp.status_code == 202
        assert resp.json()["task_id"] == "task-report-1"

    def test_interview_report_missing_session_id(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.post("/interview/report?user_id=1")
        assert resp.status_code == 400

    @patch("app.tasks.application_tasks.batch_status_check")
    def test_batch_update_endpoint(self, mock_task):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_async = MagicMock()
        mock_async.id = "task-batch-1"
        mock_task.delay.return_value = mock_async

        client = TestClient(app)
        resp = client.post("/applications/batch-update?user_id=1")
        assert resp.status_code == 202
        assert resp.json()["task_id"] == "task-batch-1"

    @patch("app.tasks.interview_tasks.generate_interview_report")
    def test_interview_report_celery_down_fallback(self, mock_task):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_task.delay.side_effect = ConnectionError("Redis down")

        client = TestClient(app)
        resp = client.post("/interview/report?user_id=1&session_id=sess-1")
        assert resp.status_code == 202
        assert "unavailable" in resp.json()["message"].lower()
