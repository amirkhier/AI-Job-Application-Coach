"""
Tests for LangChain tool wrappers.

Validates that the tools in app/tools/ correctly wrap agent methods
and are properly registered as LangChain Tool objects.
"""

import json
import pytest
from langchain_core.tools import BaseTool

from app.tools.resume_tools import (
    RESUME_TOOLS,
    analyze_resume,
    suggest_resume_improvements,
)
from app.tools.interview_tools import (
    INTERVIEW_TOOLS,
    generate_interview_questions,
    evaluate_interview_answer,
    generate_interview_summary,
)
from app.tools import ALL_TOOLS
from tests.conftest import SAMPLE_RESUME, SAMPLE_JOB_DESCRIPTION


# ================================================================ #
#  Tool registration
# ================================================================ #

class TestToolRegistration:
    """Verify tools are proper LangChain Tool instances."""

    def test_resume_tools_count(self):
        assert len(RESUME_TOOLS) == 2

    def test_interview_tools_count(self):
        assert len(INTERVIEW_TOOLS) == 3

    def test_all_tools_combined(self):
        assert len(ALL_TOOLS) == 5

    def test_all_are_base_tool_instances(self):
        for t in ALL_TOOLS:
            assert isinstance(t, BaseTool), f"{t} is not a BaseTool"

    def test_tools_have_names(self):
        names = {t.name for t in ALL_TOOLS}
        assert "analyze_resume" in names
        assert "suggest_resume_improvements" in names
        assert "generate_interview_questions" in names
        assert "evaluate_interview_answer" in names
        assert "generate_interview_summary" in names

    def test_tools_have_descriptions(self):
        for t in ALL_TOOLS:
            assert len(t.description) > 10, f"Tool {t.name} has empty/short description"


# ================================================================ #
#  Resume tools invocation
# ================================================================ #

class TestResumeToolInvocation:
    """Invoke resume tools and validate JSON output."""

    def test_analyze_resume_returns_json(self):
        result_str = analyze_resume.invoke({
            "resume_text": SAMPLE_RESUME,
            "job_description": SAMPLE_JOB_DESCRIPTION,
        })
        result = json.loads(result_str)
        assert "overall_score" in result
        assert isinstance(result["overall_score"], (int, float))

    def test_suggest_improvements_returns_json(self):
        result_str = suggest_resume_improvements.invoke({
            "resume_text": SAMPLE_RESUME,
            "job_description": SAMPLE_JOB_DESCRIPTION,
        })
        result = json.loads(result_str)
        assert "improved_summary" in result


# ================================================================ #
#  Interview tools invocation
# ================================================================ #

class TestInterviewToolInvocation:
    """Invoke interview tools and validate output."""

    def test_generate_questions_returns_json_array(self):
        result_str = generate_interview_questions.invoke({
            "role": "Backend Engineer",
            "level": "mid",
            "count": 2,
        })
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_evaluate_answer_returns_json(self):
        question = {
            "id": "q1",
            "question": "Tell me about a project you led.",
            "type": "behavioral",
            "key_points": ["Leadership", "Impact"],
        }
        result_str = evaluate_interview_answer.invoke({
            "question_json": json.dumps(question),
            "answer": "I led a migration project that improved performance by 50%.",
            "role": "Engineer",
            "level": "senior",
        })
        result = json.loads(result_str)
        assert "overall_score" in result
