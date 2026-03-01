"""
Async interview performance report — Celery task.

Generates a comprehensive interview performance report by aggregating
all Q&A pairs from a session, scoring them, and producing an
improvement plan.
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
        pass


@celery_app.task(base=BaseTaskWithRetry, bind=True, name="interview.performance_report")
def generate_interview_report(self, user_id: int, session_id: str) -> dict:
    """
    Generate a comprehensive interview performance report:
      1. Load session data (questions + answers)
      2. Per-question scoring breakdown
      3. Pattern analysis (strengths across questions)
      4. Personalised improvement plan
      5. Recommended practice questions for weak areas
    """
    from app.tools.database import DatabaseManager
    from app.agents.interview import InterviewAgent

    total_steps = 5
    start = time.time()

    # ── Step 1: Load session ─────────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 1, "total": total_steps, "detail": "Loading session data"},
    )
    db = DatabaseManager()
    try:
        db.connect()
        session = db.get_interview_session(session_id)
    except Exception as e:
        logger.error("Failed to load session %s: %s", session_id, e)
        return {"error": f"Session not found: {str(e)}", "audit_complete": False}
    finally:
        try:
            db.disconnect()
        except Exception:
            pass

    if not session:
        return {"error": f"Session {session_id} not found", "audit_complete": False}

    questions = session.get("questions", [])
    answers = session.get("answers", [])
    role = session.get("role", "Software Engineer")
    level = session.get("level", "mid")

    # ── Step 2: Per-question scoring ─────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 2, "total": total_steps, "detail": "Per-question analysis"},
    )
    per_question = []
    scores = []
    for answer in answers:
        eval_data = answer.get("evaluation", {})
        score = eval_data.get("overall_score", 0)
        scores.append(score)
        per_question.append({
            "question_id": answer.get("question_id"),
            "score": score,
            "strengths": eval_data.get("strength_areas", []),
            "improvements": eval_data.get("improvement_areas", []),
        })

    avg_score = round(sum(scores) / len(scores), 2) if scores else 0

    # ── Step 3: Pattern analysis ─────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 3, "total": total_steps, "detail": "Pattern analysis"},
    )
    all_strengths = []
    all_weaknesses = []
    for pq in per_question:
        all_strengths.extend(pq["strengths"])
        all_weaknesses.extend(pq["improvements"])

    # Count recurring themes
    def _top_items(items, n=5):
        from collections import Counter
        return [item for item, _ in Counter(items).most_common(n)]

    top_strengths = _top_items(all_strengths)
    top_weaknesses = _top_items(all_weaknesses)

    # Performance level
    if avg_score >= 8:
        performance_level = "excellent"
    elif avg_score >= 6:
        performance_level = "good"
    elif avg_score >= 4:
        performance_level = "needs_improvement"
    else:
        performance_level = "below_expectations"

    # ── Step 4: Improvement plan ─────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 4, "total": total_steps, "detail": "Generating improvement plan"},
    )
    improvement_plan = []
    if "STAR" in " ".join(all_weaknesses).upper() or avg_score < 6:
        improvement_plan.append("Practice the STAR method for structuring behavioral answers")
    if len(top_weaknesses) > 0:
        improvement_plan.append(f"Focus on improving: {', '.join(top_weaknesses[:3])}")
    if avg_score < 7:
        improvement_plan.append("Prepare specific examples with quantifiable results")
    improvement_plan.append(f"Target {role} ({level}) interview questions for continued practice")

    # ── Step 5: Report assembly ──────────────────────────────────────
    _safe_update_state(
        self,
        state="PROGRESS",
        meta={"step": 5, "total": total_steps, "detail": "Assembling report"},
    )
    elapsed = round(time.time() - start, 2)

    report = {
        "report_complete": True,
        "processing_time": elapsed,
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "level": level,
        # Summary
        "overview": {
            "total_questions": len(questions),
            "total_answered": len(answers),
            "average_score": avg_score,
            "performance_level": performance_level,
        },
        # Details
        "per_question_breakdown": per_question,
        "strength_analysis": top_strengths,
        "improvement_areas": top_weaknesses,
        "improvement_plan": improvement_plan,
    }
    logger.info(
        "Interview report for session %s complete in %.2fs (avg score: %.1f)",
        session_id, elapsed, avg_score,
    )
    return report
