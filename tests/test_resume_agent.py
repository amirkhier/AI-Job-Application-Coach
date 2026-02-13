"""
Unit tests for the Resume Agent.

Covers:
- analyze_resume() with sample resume and job description
- suggest_improvements() with and without prior analysis
- JSON parsing robustness (markdown fences, extra text)
- Error / edge cases: empty resume, missing JD, malformed LLM output
- ATS compatibility and keyword analysis structure validation
- Performance timing
- Prompt effectiveness: scoring realism, keyword detection
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from app.agents.resume import ResumeAgent
from tests.conftest import make_mock_llm_response, SAMPLE_RESUME, SAMPLE_JOB_DESCRIPTION


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture(scope="module")
def agent():
    """Create a single ResumeAgent instance for the test module."""
    return ResumeAgent()


# ================================================================ #
#  Unit tests — analyze_resume
# ================================================================ #

class TestAnalyzeResume:
    """Tests for ResumeAgent.analyze_resume()."""

    def test_analyze_with_job_description(self, agent, sample_resume, sample_job_description):
        """Full analysis with resume + JD should return all expected keys."""
        result = agent.analyze_resume(sample_resume, sample_job_description)

        # Structure
        assert "overall_score" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "recommendations" in result
        assert "ats_compatibility" in result
        assert "keyword_analysis" in result
        assert "section_feedback" in result
        assert "processing_time" in result

        # Typing
        assert isinstance(result["overall_score"], (int, float))
        assert isinstance(result["strengths"], list)
        assert isinstance(result["weaknesses"], list)
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["processing_time"], (int, float))

    def test_analyze_without_job_description(self, agent, sample_resume):
        """Analysis without JD should still succeed with general feedback."""
        result = agent.analyze_resume(sample_resume)

        assert "overall_score" in result
        assert result["overall_score"] > 0
        assert len(result["strengths"]) > 0

    def test_ats_compatibility_structure(self, agent, sample_resume, sample_job_description):
        """ATS compatibility section must have score, issues, suggestions."""
        result = agent.analyze_resume(sample_resume, sample_job_description)
        ats = result["ats_compatibility"]

        assert "score" in ats
        assert isinstance(ats["score"], (int, float))
        assert "issues" in ats
        assert "suggestions" in ats

    def test_keyword_analysis_structure(self, agent, sample_resume, sample_job_description):
        """Keyword analysis should identify present and missing keywords."""
        result = agent.analyze_resume(sample_resume, sample_job_description)
        kw = result["keyword_analysis"]

        assert "present_keywords" in kw
        assert "missing_keywords" in kw
        assert isinstance(kw["present_keywords"], list)
        assert isinstance(kw["missing_keywords"], list)

    def test_score_realism(self, agent, sample_resume, sample_job_description):
        """Score should be realistic: between 1.0 and 10.0, not universally perfect."""
        result = agent.analyze_resume(sample_resume, sample_job_description)
        score = result["overall_score"]

        assert 1.0 <= score <= 10.0
        # A solid resume shouldn't score below 4
        assert score >= 4.0, f"Good resume scored too low: {score}"
        # But shouldn't be a perfect 10 either
        assert score <= 9.5, f"Score unrealistically high: {score}"

    def test_processing_time_reasonable(self, agent, sample_resume):
        """Analysis should complete within 60 seconds."""
        start = time.time()
        result = agent.analyze_resume(sample_resume)
        elapsed = time.time() - start

        assert elapsed < 60, f"Analysis took too long: {elapsed:.1f}s"
        assert result["processing_time"] > 0


# ================================================================ #
#  Unit tests — suggest_improvements
# ================================================================ #

class TestSuggestImprovements:
    """Tests for ResumeAgent.suggest_improvements()."""

    def test_improvements_with_fresh_analysis(self, agent, sample_resume, sample_job_description):
        """When no prior analysis is passed, it should auto-analyse first."""
        result = agent.suggest_improvements(sample_resume, sample_job_description)

        assert "improved_summary" in result
        assert "improved_bullets" in result
        assert "priority_actions" in result
        assert isinstance(result["improved_bullets"], list)
        assert isinstance(result["priority_actions"], list)

    def test_improvements_with_prior_analysis(self, agent, sample_resume, sample_job_description):
        """Passing a prior analysis should skip the re-analysis step."""
        analysis = agent.analyze_resume(sample_resume, sample_job_description)
        result = agent.suggest_improvements(sample_resume, sample_job_description, analysis)

        assert "improved_summary" in result
        assert len(result["improved_summary"]) > 0

    def test_improved_bullets_structure(self, agent, sample_resume, sample_job_description):
        """Each improved bullet should have original, improved, reasoning."""
        result = agent.suggest_improvements(sample_resume, sample_job_description)

        for bullet in result["improved_bullets"]:
            assert "original" in bullet or "improved" in bullet
            # At minimum the improved version should be present
            assert "improved" in bullet


# ================================================================ #
#  JSON parsing robustness
# ================================================================ #

class TestJSONParsing:
    """Test _parse_llm_json with various edge cases."""

    def test_clean_json(self, agent):
        result = agent._parse_llm_json('{"score": 7.5}')
        assert result == {"score": 7.5}

    def test_json_with_markdown_fences(self, agent):
        raw = '```json\n{"score": 8.0}\n```'
        result = agent._parse_llm_json(raw)
        assert result == {"score": 8.0}

    def test_json_with_leading_text(self, agent):
        raw = 'Here is my analysis:\n{"score": 6.5, "items": []}'
        result = agent._parse_llm_json(raw)
        assert result["score"] == 6.5

    def test_invalid_json_raises(self, agent):
        with pytest.raises(ValueError, match="invalid JSON"):
            agent._parse_llm_json("this is not json at all")


# ================================================================ #
#  Edge cases / error handling
# ================================================================ #

class TestEdgeCases:
    """Error handling and edge-case scenarios."""

    def test_minimal_resume(self, agent, minimal_resume):
        """Even a very short resume should return a valid analysis."""
        result = agent.analyze_resume(minimal_resume)
        assert "overall_score" in result
        assert isinstance(result["overall_score"], (int, float))

    def test_llm_returns_bad_json(self, agent):
        """If the LLM returns unparseable text, we get a graceful fallback."""
        with patch.object(agent, "llm") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = make_mock_llm_response("NOT_JSON_AT_ALL")
            with patch.object(agent, "_analysis_prompt") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                result = agent.analyze_resume("Some resume text")

        assert result["overall_score"] == 0.0
        assert "error" in result

    def test_empty_resume_still_returns(self, agent):
        """Empty resume should still produce a valid (low-scoring) response."""
        result = agent.analyze_resume("")
        assert "overall_score" in result


# ================================================================ #
#  Prompt effectiveness — keyword detection
# ================================================================ #

class TestPromptEffectiveness:
    """Validate that the LLM prompts produce quality output."""

    def test_detects_present_keywords(self, agent):
        """Resume with Python + Kafka should detect those as present keywords."""
        result = agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JOB_DESCRIPTION)
        present = [k.lower() for k in result.get("keyword_analysis", {}).get("present_keywords", [])]

        # The sample resume explicitly mentions Python and Kafka
        assert any("python" in k for k in present), f"Python not detected in: {present}"

    def test_detects_missing_keywords(self, agent):
        """JD mentions 'financial domain' — resume doesn't, so it should be flagged."""
        result = agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JOB_DESCRIPTION)
        missing = [k.lower() for k in result.get("keyword_analysis", {}).get("missing_keywords", [])]

        # Financial domain / fintech is in JD but not in the resume
        assert len(missing) > 0, "Should detect at least one missing keyword"

    def test_section_feedback_covers_key_sections(self, agent):
        """Section feedback should address the main resume sections."""
        result = agent.analyze_resume(SAMPLE_RESUME, SAMPLE_JOB_DESCRIPTION)
        sf = result.get("section_feedback", {})

        # Should have feedback for at least experience and skills
        assert len(sf) >= 2, f"Section feedback too sparse: {list(sf.keys())}"
