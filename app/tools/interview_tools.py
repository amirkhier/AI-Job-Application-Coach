"""
LangChain tool wrappers for the Interview Agent.

These tools expose InterviewAgent methods as formal LangChain ``Tool`` objects
so they can be invoked by the LangGraph orchestration layer (Phase 3).
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from app.agents.interview import InterviewAgent

logger = logging.getLogger(__name__)

# Module-level singleton
_interview_agent: Optional[InterviewAgent] = None


def _get_agent() -> InterviewAgent:
    global _interview_agent
    if _interview_agent is None:
        _interview_agent = InterviewAgent()
    return _interview_agent


@tool
def generate_interview_questions(
    role: str,
    level: str = "mid",
    count: int = 5,
) -> str:
    """Generate role-specific interview questions.

    Args:
        role: Target job title (e.g. "Backend Engineer").
        level: Seniority level — junior, mid, senior, or lead.
        count: Number of questions to generate (1–10).

    Returns:
        JSON string containing an array of question objects,
        each with id, question, type, difficulty, and key_points.
    """
    agent = _get_agent()
    questions = agent.generate_questions(role, level, count)
    return json.dumps(questions, indent=2)


@tool
def evaluate_interview_answer(
    question_json: str,
    answer: str,
    role: str = "Software Engineer",
    level: str = "mid",
) -> str:
    """Evaluate a candidate's answer to an interview question.

    Args:
        question_json: JSON string of the question dict (must have
            'question', 'type', 'key_points').
        answer: The candidate's answer text.
        role: Target role for evaluation context.
        level: Seniority level for evaluation context.

    Returns:
        JSON string with overall_score, dimension_scores,
        strength_areas, improvement_areas, specific_feedback,
        and suggested_improvement.
    """
    agent = _get_agent()
    question = json.loads(question_json)
    result = agent.evaluate_answer(question, answer, role, level)
    return json.dumps(result, indent=2)


@tool
def generate_interview_summary(
    questions_json: str,
    answers_with_feedback_json: str,
    role: str = "Software Engineer",
    level: str = "mid",
) -> str:
    """Generate a comprehensive summary for a completed interview session.

    Args:
        questions_json: JSON array of question dicts.
        answers_with_feedback_json: JSON array of dicts with question_id,
            answer, and evaluation.
        role: Target role.
        level: Seniority level.

    Returns:
        JSON string with overall_score, performance_level,
        strongest/weakest areas, recommendations, and next_steps.
    """
    agent = _get_agent()
    questions = json.loads(questions_json)
    answers = json.loads(answers_with_feedback_json)
    result = agent.generate_session_summary(questions, answers, role, level)
    return json.dumps(result, indent=2)


# Convenience list for registering all interview tools at once
INTERVIEW_TOOLS = [
    generate_interview_questions,
    evaluate_interview_answer,
    generate_interview_summary,
]
