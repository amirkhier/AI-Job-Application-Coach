"""
Async resume audit — multi-step Celery task for deep resume analysis.

The audit runs five steps and reports progress via Celery state updates
so the client can poll ``/tasks/{task_id}/status`` for live progress.
"""

from __future__ import annotations

import logging
import time

from app.celery_worker import celery_app
from app.tasks.base import BaseTaskWithRetry

logger = logging.getLogger(__name__)


def _safe_update_state(task, **kwargs):
    """Update Celery task state, silently skipping if backend is unavailable."""
    try:
        task.update_state(**kwargs)
    except Exception:
        pass  # Non-critical: progress reporting should not break the task


@celery_app.task(base=BaseTaskWithRetry, bind=True, name="resume.detailed_audit")
def detailed_resume_audit(
    self,
    resume_text: str,
    job_description: str = None,
    user_id: int = 1,
) -> dict:
    """
    Multi-step resume audit:
      1. Basic analysis (score, strengths, weaknesses)
      2. Improvement suggestions (BAR bullet rewrites)
      3. Skill extraction & gap analysis (vs. job description)
      4. Section-by-section deep dive
      5. Final comprehensive report assembly
    """
    from app.agents.resume import ResumeAgent

    agent = ResumeAgent()
    total_steps = 5
    start = time.time()

    # ── Step 1: Basic analysis ───────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 1, "total": total_steps, "detail": "Basic analysis"},
    )
    analysis = agent.analyze_resume(resume_text, job_description or "")
    logger.info("Audit step 1/5 complete for user %s", user_id)

    # ── Step 2: Improvement suggestions ──────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 2, "total": total_steps, "detail": "Improvement suggestions"},
    )
    improvements = agent.suggest_improvements(resume_text, job_description or "")
    logger.info("Audit step 2/5 complete for user %s", user_id)

    # ── Step 3: Skill extraction & gap analysis ──────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 3, "total": total_steps, "detail": "Skill gap analysis"},
    )
    # Use the keyword analysis from step 1 + enrich
    keyword_data = analysis.get("keyword_analysis", {})
    skill_gap = {
        "present_skills": keyword_data.get("present_keywords", []),
        "missing_skills": keyword_data.get("missing_keywords", []),
        "gap_severity": (
            "high" if len(keyword_data.get("missing_keywords", [])) > 5
            else "medium" if len(keyword_data.get("missing_keywords", [])) > 2
            else "low"
        ),
    }
    logger.info("Audit step 3/5 complete for user %s", user_id)

    # ── Step 4: Section-by-section deep dive ─────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 4, "total": total_steps, "detail": "Section analysis"},
    )
    section_feedback = analysis.get("section_feedback", {})
    ats_data = analysis.get("ats_compatibility", {})
    logger.info("Audit step 4/5 complete for user %s", user_id)

    # ── Step 5: Report assembly ──────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 5, "total": total_steps, "detail": "Report assembly"},
    )
    elapsed = round(time.time() - start, 2)

    report = {
        "audit_complete": True,
        "processing_time": elapsed,
        "user_id": user_id,
        # Core analysis
        "overall_score": analysis.get("overall_score", 0.0),
        "strengths": analysis.get("strengths", []),
        "weaknesses": analysis.get("weaknesses", []),
        "recommendations": analysis.get("recommendations", []),
        # ATS
        "ats_compatibility": ats_data,
        # Skills
        "skill_gap_analysis": skill_gap,
        # Sections
        "section_feedback": section_feedback,
        # Improvements
        "improved_summary": improvements.get("improved_summary", ""),
        "improved_bullets": improvements.get("improved_bullets", []),
        "priority_actions": improvements.get("priority_actions", []),
    }
    logger.info("Audit complete for user %s in %.2fs", user_id, elapsed)
    return report
