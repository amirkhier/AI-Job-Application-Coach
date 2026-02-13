"""
Unit tests for the Interview Agent.

Covers:
- generate_questions() for different roles and levels
- evaluate_answer() with good, mediocre, and weak answers
- generate_session_summary() for completed sessions
- JSON parsing robustness
- Error handling and fallback behaviour
- Performance timing
- Prompt effectiveness: question diversity, score differentiation
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from app.agents.interview import InterviewAgent
from tests.conftest import make_mock_llm_response


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture(scope="module")
def agent():
    return InterviewAgent()


# ================================================================ #
#  Question generation
# ================================================================ #

class TestGenerateQuestions:
    """Tests for InterviewAgent.generate_questions()."""

    def test_generates_correct_count(self, agent):
        """Should return the requested number of questions."""
        questions = agent.generate_questions("Backend Engineer", "senior", count=3)
        assert len(questions) == 3

    def test_question_structure(self, agent):
        """Each question must have id, question, type, difficulty, key_points."""
        questions = agent.generate_questions("Data Scientist", "mid", count=3)

        for q in questions:
            assert "id" in q
            assert "question" in q
            assert "type" in q
            assert "difficulty" in q
            assert "key_points" in q
            assert isinstance(q["question"], str)
            assert len(q["question"]) > 10
            assert isinstance(q["key_points"], list)

    def test_question_type_diversity(self, agent):
        """With 5+ questions, there should be at least 2 different types."""
        questions = agent.generate_questions("Software Engineer", "senior", count=5)
        types = {q["type"] for q in questions}

        assert len(types) >= 2, f"Only one question type found: {types}"

    def test_different_roles_produce_different_questions(self, agent):
        """Questions for a Data Engineer should differ from a QA Engineer."""
        de_questions = agent.generate_questions("Data Engineer", "mid", count=3)
        qa_questions = agent.generate_questions("QA Engineer", "mid", count=3)

        de_texts = {q["question"] for q in de_questions}
        qa_texts = {q["question"] for q in qa_questions}

        # They shouldn't be identical sets
        assert de_texts != qa_texts, "Different roles produced identical questions"

    def test_fallback_on_llm_failure(self, agent):
        """If the LLM fails, we should get sensible fallback questions."""
        with patch.object(agent, "llm") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("API error")
            with patch.object(agent, "_question_prompt") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                questions = agent.generate_questions("Engineer", "mid", count=3)

        assert len(questions) == 3
        assert all("question" in q for q in questions)

    def test_performance_question_generation(self, agent):
        """Question generation should complete within 60 seconds."""
        start = time.time()
        agent.generate_questions("Python Developer", "junior", count=3)
        elapsed = time.time() - start
        assert elapsed < 60, f"Question generation took {elapsed:.1f}s"


# ================================================================ #
#  Answer evaluation
# ================================================================ #

class TestEvaluateAnswer:
    """Tests for InterviewAgent.evaluate_answer()."""

    def test_evaluation_structure(self, agent, sample_question, sample_answer):
        """Evaluation must return all expected keys."""
        result = agent.evaluate_answer(sample_question, sample_answer)

        assert "overall_score" in result
        assert "dimension_scores" in result
        assert "strength_areas" in result
        assert "improvement_areas" in result
        assert "specific_feedback" in result
        assert "suggested_improvement" in result
        assert "processing_time" in result

    def test_dimension_scores(self, agent, sample_question, sample_answer):
        """Dimension scores must include all 4 dimensions."""
        result = agent.evaluate_answer(sample_question, sample_answer)
        dims = result["dimension_scores"]

        for key in ("relevance", "depth", "structure", "communication"):
            assert key in dims, f"Missing dimension: {key}"
            assert 1.0 <= dims[key] <= 10.0

    def test_good_answer_scores_higher_than_weak(self, agent, sample_question, sample_answer, weak_answer):
        """A detailed answer should score higher than a vague one-liner."""
        good_result = agent.evaluate_answer(sample_question, sample_answer)
        weak_result = agent.evaluate_answer(sample_question, weak_answer)

        assert good_result["overall_score"] > weak_result["overall_score"], (
            f"Good answer ({good_result['overall_score']}) should score higher "
            f"than weak answer ({weak_result['overall_score']})"
        )

    def test_score_range(self, agent, sample_question, sample_answer):
        """Scores should be in the 1.0–10.0 range."""
        result = agent.evaluate_answer(sample_question, sample_answer)
        assert 1.0 <= result["overall_score"] <= 10.0

    def test_feedback_is_substantive(self, agent, sample_question, sample_answer):
        """Specific feedback should be more than a few words."""
        result = agent.evaluate_answer(sample_question, sample_answer)
        assert len(result["specific_feedback"]) > 20, "Feedback too short"

    def test_evaluation_fallback_on_parse_error(self, agent, sample_question):
        """If LLM returns bad JSON, we get a structured fallback."""
        with patch.object(agent, "llm") as mock_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = make_mock_llm_response("BROKEN!")
            with patch.object(agent, "_evaluation_prompt") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                result = agent.evaluate_answer(sample_question, "test answer")

        assert result["overall_score"] == 5.0
        assert "error" in result


# ================================================================ #
#  Session summary
# ================================================================ #

class TestSessionSummary:
    """Tests for InterviewAgent.generate_session_summary()."""

    def test_summary_structure(self, agent):
        """Full session summary should have all required keys."""
        questions = agent.generate_questions("Engineer", "mid", count=2)

        answers = []
        for q in questions:
            evaluation = agent.evaluate_answer(q, "I have some experience with that.")
            answers.append({
                "question_id": q["id"],
                "answer": "I have some experience with that.",
                "evaluation": evaluation,
            })

        summary = agent.generate_session_summary(questions, answers, "Engineer", "mid")

        assert "overall_score" in summary
        assert "performance_level" in summary
        assert "key_recommendations" in summary
        assert isinstance(summary["key_recommendations"], list)


# ================================================================ #
#  JSON parsing
# ================================================================ #

class TestJSONParsing:

    def test_parse_array(self):
        raw = '[{"id": "q1", "question": "test?"}]'
        result = InterviewAgent._parse_llm_json(raw)
        assert isinstance(result, list)

    def test_parse_with_fences(self):
        raw = '```json\n[{"id": "q1"}]\n```'
        result = InterviewAgent._parse_llm_json(raw)
        assert isinstance(result, list)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            InterviewAgent._parse_llm_json("not json")


# ================================================================ #
#  Helper functions
# ================================================================ #

class TestHelpers:

    def test_score_to_level_exceptional(self):
        assert InterviewAgent._score_to_level(9.5) == "exceptional"

    def test_score_to_level_strong(self):
        assert InterviewAgent._score_to_level(8.0) == "strong"

    def test_score_to_level_competent(self):
        assert InterviewAgent._score_to_level(6.5) == "competent"

    def test_score_to_level_developing(self):
        assert InterviewAgent._score_to_level(4.5) == "developing"

    def test_score_to_level_needs_improvement(self):
        assert InterviewAgent._score_to_level(2.0) == "needs_improvement"
