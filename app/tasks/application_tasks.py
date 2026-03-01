"""
Async application tracking tasks — Celery tasks for batch operations.

Provides:
  - Batch status check (flag stale / overdue applications)
  - Follow-up reminder generation
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app.celery_worker import celery_app
from app.tasks.base import BaseTaskWithRetry

logger = logging.getLogger(__name__)


@celery_app.task(base=BaseTaskWithRetry, bind=True, name="applications.batch_status_check")
def batch_status_check(self, user_id: int) -> dict:
    """
    For all active applications belonging to *user_id*:
      - Check if follow_up_date is overdue
      - Flag stale applications (no activity > 30 days)
      - Return a summary of items needing attention
    """
    from app.tools.database import DatabaseManager

    db = DatabaseManager()
    try:
        db.connect()
        applications = db.get_applications(user_id=user_id)
    except Exception as e:
        logger.error("Batch status check failed for user %s: %s", user_id, e)
        return {"error": str(e), "check_complete": False}
    finally:
        try:
            db.disconnect()
        except Exception:
            pass

    today = date.today()
    stale_threshold = today - timedelta(days=30)
    terminal = {"offer", "rejected", "withdrawn"}

    overdue_follow_ups = []
    stale_applications = []
    active_count = 0

    for app_record in applications:
        if app_record["status"] in terminal:
            continue
        active_count += 1

        # Overdue follow-up
        fud = app_record.get("follow_up_date")
        if fud and fud <= today:
            overdue_follow_ups.append({
                "id": app_record["id"],
                "company": app_record["company_name"],
                "position": app_record["position_title"],
                "follow_up_date": str(fud),
                "days_overdue": (today - fud).days,
            })

        # Stale (no activity > 30 days)
        updated = app_record.get("updated_at")
        if updated:
            updated_date = updated.date() if hasattr(updated, "date") else updated
            if updated_date < stale_threshold:
                stale_applications.append({
                    "id": app_record["id"],
                    "company": app_record["company_name"],
                    "position": app_record["position_title"],
                    "last_activity": str(updated_date),
                    "days_stale": (today - updated_date).days,
                })

    result = {
        "check_complete": True,
        "user_id": user_id,
        "total_applications": len(applications),
        "active_applications": active_count,
        "overdue_follow_ups": overdue_follow_ups,
        "stale_applications": stale_applications,
        "needs_attention": len(overdue_follow_ups) + len(stale_applications),
    }
    logger.info(
        "Batch check for user %s: %d overdue, %d stale out of %d active",
        user_id, len(overdue_follow_ups), len(stale_applications), active_count,
    )
    return result


@celery_app.task(name="applications.generate_follow_up_reminders")
def generate_follow_up_reminders(user_id: int) -> dict:
    """
    Generate follow-up reminder messages for all overdue applications.
    Returns a list of reminder objects the front-end can display / email.
    """
    from app.tools.database import DatabaseManager

    db = DatabaseManager()
    try:
        db.connect()
        applications = db.get_applications(user_id=user_id)
    except Exception as e:
        logger.error("Follow-up reminder generation failed: %s", e)
        return {"error": str(e), "reminders": []}
    finally:
        try:
            db.disconnect()
        except Exception:
            pass

    today = date.today()
    terminal = {"offer", "rejected", "withdrawn"}
    reminders = []

    for app_record in applications:
        fud = app_record.get("follow_up_date")
        if not fud or app_record["status"] in terminal:
            continue
        if fud <= today:
            days_overdue = (today - fud).days
            urgency = "high" if days_overdue > 7 else "medium" if days_overdue > 3 else "low"
            reminders.append({
                "application_id": app_record["id"],
                "company": app_record["company_name"],
                "position": app_record["position_title"],
                "follow_up_date": str(fud),
                "days_overdue": days_overdue,
                "urgency": urgency,
                "message": (
                    f"Follow up on your {app_record['position_title']} application "
                    f"at {app_record['company_name']} — {days_overdue} day(s) overdue."
                ),
            })

    return {"reminders": reminders, "count": len(reminders)}
