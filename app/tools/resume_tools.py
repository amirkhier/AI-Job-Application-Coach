"""
LangChain tool wrappers for the Resume Agent.

These tools expose ResumeAgent methods as formal LangChain ``Tool`` objects
so they can be invoked by the LangGraph orchestration layer (Phase 3).
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from app.agents.resume import ResumeAgent

logger = logging.getLogger(__name__)

# Module-level singleton — created lazily on first tool invocation
_resume_agent: Optional[ResumeAgent] = None


def _get_agent() -> ResumeAgent:
    global _resume_agent
    if _resume_agent is None:
        _resume_agent = ResumeAgent()
    return _resume_agent


@tool
def analyze_resume(resume_text: str, job_description: str = "") -> str:
    """Analyze a resume and return structured feedback including overall score,
    strengths, weaknesses, ATS compatibility, keyword analysis, and section feedback.

    Args:
        resume_text: The full plain-text content of the resume.
        job_description: Optional target job description for tailored analysis.

    Returns:
        JSON string with analysis results.
    """
    agent = _get_agent()
    result = agent.analyze_resume(resume_text, job_description)
    return json.dumps(result, indent=2)


@tool
def suggest_resume_improvements(
    resume_text: str,
    job_description: str = "",
    analysis_json: str = "",
) -> str:
    """Generate concrete improvement suggestions for a resume including
    rewritten bullet points, an improved summary, and priority actions.

    Args:
        resume_text: The full plain-text content of the resume.
        job_description: Optional job description for targeted improvements.
        analysis_json: Optional JSON string of a previous analysis. If empty,
            a fresh analysis is performed automatically.

    Returns:
        JSON string with improvement suggestions.
    """
    agent = _get_agent()
    analysis = json.loads(analysis_json) if analysis_json else None
    result = agent.suggest_improvements(resume_text, job_description, analysis)
    return json.dumps(result, indent=2)


# Convenience list for registering all resume tools at once
RESUME_TOOLS = [analyze_resume, suggest_resume_improvements]
