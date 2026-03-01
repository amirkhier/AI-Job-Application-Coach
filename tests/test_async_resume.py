"""
Tests for Phase 4 Step 5 — Async Resume Audit.

Covers:
  - Resume audit Celery task in eager mode
  - Task progress updates
  - /resume/audit endpoint dispatching
  - Graceful fallback when Celery is unavailable
"""

import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# Resume Audit Task (eager mode)
# ===========================================================================

class TestResumeAuditTask:
    """Test the detailed_resume_audit Celery task."""

    @patch("app.agents.resume.ResumeAgent")
    def test_audit_task_returns_report(self, MockAgent):
        """Run the audit task in eager mode and verify report structure."""
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_agent = MockAgent.return_value
        mock_agent.analyze_resume.return_value = {
            "overall_score": 7.5,
            "strengths": ["Strong Python skills"],
            "weaknesses": ["Missing leadership examples"],
            "recommendations": ["Add metrics"],
            "ats_compatibility": {"score": 8.0, "issues": [], "suggestions": []},
            "keyword_analysis": {
                "present_keywords": ["Python", "FastAPI"],
                "missing_keywords": ["Docker", "Kubernetes"],
            },
            "section_feedback": {},
        }
        mock_agent.suggest_improvements.return_value = {
            "improved_summary": "Enhanced summary",
            "improved_bullets": [],
            "priority_actions": ["Add Docker experience"],
        }

        try:
            from app.tasks.resume_tasks import detailed_resume_audit

            result = detailed_resume_audit.delay(
                resume_text="Test resume " * 20,
                job_description="Backend developer",
                user_id=1,
            )
            report = result.get()
            assert report["audit_complete"] is True
            assert report["overall_score"] == 7.5
            assert "skill_gap_analysis" in report
            assert report["skill_gap_analysis"]["gap_severity"] in ("low", "medium", "high")
            assert "priority_actions" in report
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)

    @patch("app.agents.resume.ResumeAgent")
    def test_audit_task_gap_severity_high(self, MockAgent):
        """Gap severity should be 'high' when many skills are missing."""
        from app.celery_worker import celery_app
        celery_app.conf.update(task_always_eager=True, task_eager_propagates=True, result_backend='cache+memory://', broker_url='memory://')

        mock_agent = MockAgent.return_value
        mock_agent.analyze_resume.return_value = {
            "overall_score": 4.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "ats_compatibility": {},
            "keyword_analysis": {
                "present_keywords": ["Python"],
                "missing_keywords": ["A", "B", "C", "D", "E", "F"],  # 6 missing
            },
            "section_feedback": {},
        }
        mock_agent.suggest_improvements.return_value = {
            "improved_summary": "",
            "improved_bullets": [],
            "priority_actions": [],
        }

        try:
            from app.tasks.resume_tasks import detailed_resume_audit

            result = detailed_resume_audit.delay(resume_text="x " * 50)
            report = result.get()
            assert report["skill_gap_analysis"]["gap_severity"] == "high"
        finally:
            celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ===========================================================================
# /resume/audit Endpoint
# ===========================================================================

class TestResumeAuditEndpoint:
    """Test the endpoint that dispatches the Celery task."""

    @patch("app.tasks.resume_tasks.detailed_resume_audit")
    def test_endpoint_dispatches_task(self, mock_task):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_async = MagicMock()
        mock_async.id = "celery-task-abc"
        mock_task.delay.return_value = mock_async

        client = TestClient(app)
        resp = client.post(
            "/resume/audit",
            json={"resume_text": "x " * 50, "user_id": 1},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["task_id"] == "celery-task-abc"
        assert data["status"] == "queued"

    @patch("app.tasks.resume_tasks.detailed_resume_audit")
    def test_endpoint_fallback_when_celery_unavailable(self, mock_task):
        """When Celery is down, the endpoint should still return 202 with a placeholder."""
        from fastapi.testclient import TestClient
        from app.main import app

        mock_task.delay.side_effect = ConnectionError("Redis refused")

        client = TestClient(app)
        resp = client.post(
            "/resume/audit",
            json={"resume_text": "x " * 50, "user_id": 1},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert "unavailable" in data["message"].lower()
