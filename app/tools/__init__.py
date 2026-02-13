"""Tools package for the AI Job Application Coach.

Exposes LangChain tool wrappers for each agent, ready for LangGraph orchestration.
"""

from app.tools.resume_tools import RESUME_TOOLS, analyze_resume, suggest_resume_improvements
from app.tools.interview_tools import (
    INTERVIEW_TOOLS,
    generate_interview_questions,
    evaluate_interview_answer,
    generate_interview_summary,
)

ALL_TOOLS = RESUME_TOOLS + INTERVIEW_TOOLS

__all__ = [
    "RESUME_TOOLS",
    "INTERVIEW_TOOLS",
    "ALL_TOOLS",
    "analyze_resume",
    "suggest_resume_improvements",
    "generate_interview_questions",
    "evaluate_interview_answer",
    "generate_interview_summary",
]