"""
Unit tests for the Router Agent (LLM-powered intent classification).

Covers:
- LLM-based classification for each intent type
- Keyword fallback when LLM is unavailable
- Confidence thresholding and cross-checking
- Active interview session override
- Explicit new-intent detection during active sessions
- JSON parsing robustness (fences, malformed output)
- resolve_agent() mapping
- Edge cases: empty query, very long query, non-English
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.router import RouterAgent, INTENT_TO_AGENT, CONFIDENCE_THRESHOLD
from app.graph.state import INTENT_TYPES
from tests.conftest import make_mock_llm_response


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture(scope="module")
def agent():
    """Router agent using real LLM (requires OPENAI_API_KEY)."""
    return RouterAgent()


@pytest.fixture
def mock_agent():
    """Router agent with mocked LLM for deterministic tests."""
    with patch("app.agents.router.ChatOpenAI") as MockLLM:
        instance = MockLLM.return_value
        router = RouterAgent()
        router.llm = instance
        # Re-build the chain with the mocked LLM
        router.classification_chain = router.classification_prompt | instance
        yield router, instance


# ================================================================ #
#  LLM-based classification (live — requires API key)
# ================================================================ #

class TestLLMClassification:
    """Tests that exercise the real LLM for intent classification."""

    @pytest.mark.parametrize("query, expected_intent", [
        ("Can you review my resume?", "resume_analysis"),
        ("Help me improve my resume bullet points", "resume_improvement"),
        ("I want to practice for a software engineer interview", "interview_practice"),
        ("Search for Python developer jobs in Tel Aviv", "job_search"),
        ("How should I negotiate my salary?", "career_advice"),
        ("What's the status of my application to Google?", "application_tracking"),
    ])
    def test_classifies_clear_intents(self, agent, query, expected_intent):
        """LLM should correctly classify unambiguous queries."""
        result = agent.classify_intent(query)

        assert result["intent"] == expected_intent, (
            f"Expected '{expected_intent}' for '{query}', got '{result['intent']}'"
        )
        assert result["confidence"] >= 0.8
        assert result["reasoning"]
        assert result["classification_method"] in ("llm", "keyword_fallback")

    def test_response_structure(self, agent):
        """Every classification result must have the required keys."""
        result = agent.classify_intent("Tell me about interview preparation tips")

        assert "intent" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "classification_method" in result
        assert "processing_time" in result
        assert result["intent"] in INTENT_TYPES

    def test_confidence_is_bounded(self, agent):
        """Confidence must be in [0.0, 1.0]."""
        result = agent.classify_intent("Help me find a job")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_ambiguous_query_prefers_career_advice(self, agent):
        """Ambiguous queries should lean toward career_advice, not unknown."""
        result = agent.classify_intent("I need help")
        assert result["intent"] in ("career_advice", "general_question"), (
            f"Ambiguous query classified as '{result['intent']}'"
        )


# ================================================================ #
#  Keyword fallback classifier
# ================================================================ #

class TestKeywordClassifier:
    """Tests for the deterministic keyword-based fallback."""

    @pytest.mark.parametrize("query, expected_intent", [
        ("review my resume", "resume_analysis"),
        ("improve my resume bullets", "resume_improvement"),
        ("mock interview practice", "interview_practice"),
        ("start interview session", "interview_start"),
        ("search for jobs in Haifa", "job_search"),
        ("salary negotiation tips", "career_advice"),
        ("track my application status", "application_tracking"),
        ("hello world", "unknown"),
    ])
    def test_keyword_classification(self, query, expected_intent):
        result = RouterAgent._keyword_classify(query)
        assert result["intent"] == expected_intent
        assert result["classification_method"] == "keyword"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_query(self):
        result = RouterAgent._keyword_classify("")
        assert result["intent"] == "unknown"


# ================================================================ #
#  Active interview session override
# ================================================================ #

class TestSessionOverride:
    """Tests for interview session context-aware routing."""

    def test_active_session_routes_to_interview_answer(self, agent):
        """When an interview session is active, user input = answer."""
        result = agent.classify_intent(
            "I once led a team of 5 engineers to deliver a critical feature...",
            has_active_session=True,
        )
        assert result["intent"] == "interview_answer"
        assert result["confidence"] >= 0.9
        assert result["classification_method"] == "session_override"

    def test_active_session_allows_explicit_new_intent(self, agent):
        """User can break out of an interview by stating a new intent."""
        result = agent.classify_intent(
            "Stop interview, I want to search for jobs instead",
            has_active_session=True,
        )
        # Should NOT be session_override because of explicit new intent
        assert result["classification_method"] != "session_override"
        assert result["intent"] != "interview_answer"

    def test_explicit_new_intent_detection(self):
        """_is_explicit_new_intent should catch override signals."""
        assert RouterAgent._is_explicit_new_intent("I want to review my resume")
        assert RouterAgent._is_explicit_new_intent("find job openings")
        assert RouterAgent._is_explicit_new_intent("end session please")
        assert not RouterAgent._is_explicit_new_intent("Well, at my previous company I...")
        assert not RouterAgent._is_explicit_new_intent("We used Python and FastAPI")


# ================================================================ #
#  LLM fallback behaviour (mocked)
# ================================================================ #

class TestLLMFallback:
    """Tests that keyword fallback engages when LLM fails."""

    def test_falls_back_on_llm_exception(self, mock_agent):
        router, mock_llm = mock_agent
        mock_llm.invoke.side_effect = Exception("API error")

        # Re-patch the chain to raise
        router.classification_chain = MagicMock()
        router.classification_chain.invoke.side_effect = Exception("API error")

        result = router.classify_intent("review my resume")
        assert result["intent"] == "resume_analysis"
        assert result["classification_method"] == "keyword_fallback"

    def test_falls_back_on_low_confidence(self, mock_agent):
        router, mock_llm = mock_agent

        low_confidence_response = json.dumps({
            "intent": "career_advice",
            "confidence": 0.3,
            "reasoning": "Uncertain"
        })
        router.classification_chain = MagicMock()
        router.classification_chain.invoke.return_value = make_mock_llm_response(
            low_confidence_response
        )

        result = router.classify_intent("search for jobs in Tel Aviv")
        # Keyword should win because "search" + "jobs" → job_search at 0.85
        assert result["intent"] == "job_search"
        assert result["classification_method"] == "keyword_fallback"


# ================================================================ #
#  JSON parsing robustness
# ================================================================ #

class TestJSONParsing:
    """Tests for _parse_llm_json edge cases."""

    def test_clean_json(self):
        raw = '{"intent": "resume_analysis", "confidence": 0.9, "reasoning": "test"}'
        result = RouterAgent._parse_llm_json(raw)
        assert result["intent"] == "resume_analysis"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"intent": "job_search", "confidence": 0.85, "reasoning": "test"}\n```'
        result = RouterAgent._parse_llm_json(raw)
        assert result["intent"] == "job_search"

    def test_json_with_extra_text(self):
        raw = 'Here is the result:\n{"intent": "career_advice", "confidence": 0.8, "reasoning": "test"}\nDone.'
        result = RouterAgent._parse_llm_json(raw)
        assert result["intent"] == "career_advice"

    def test_completely_malformed(self):
        raw = "I don't know what to do"
        result = RouterAgent._parse_llm_json(raw)
        assert result["intent"] == "unknown"
        assert result["confidence"] == 0.0

    def test_invalid_intent_is_normalized(self, mock_agent):
        """LLM returning an intent not in INTENT_TYPES → 'unknown'."""
        router, _ = mock_agent
        router.classification_chain = MagicMock()
        router.classification_chain.invoke.return_value = make_mock_llm_response(
            json.dumps({
                "intent": "make_coffee",
                "confidence": 0.9,
                "reasoning": "Not a valid intent"
            })
        )
        result = router.classify_intent("make me a coffee")
        assert result["intent"] == "unknown"


# ================================================================ #
#  resolve_agent mapping
# ================================================================ #

class TestResolveAgent:
    """Tests for intent → agent node mapping."""

    def test_all_intents_mapped(self):
        """Every intent in INTENT_TYPES should have a mapping."""
        for intent in INTENT_TYPES:
            agent = RouterAgent.resolve_agent(intent)
            assert agent in ("resume", "interview", "job_search", "knowledge"), (
                f"Intent '{intent}' mapped to unexpected agent '{agent}'"
            )

    @pytest.mark.parametrize("intent, expected_agent", [
        ("resume_analysis", "resume"),
        ("resume_improvement", "resume"),
        ("interview_practice", "interview"),
        ("interview_start", "interview"),
        ("interview_answer", "interview"),
        ("job_search", "job_search"),
        ("career_advice", "knowledge"),
        ("application_tracking", "knowledge"),
        ("general_question", "knowledge"),
        ("unknown", "knowledge"),
    ])
    def test_specific_mappings(self, intent, expected_agent):
        assert RouterAgent.resolve_agent(intent) == expected_agent

    def test_unmapped_intent_defaults_to_knowledge(self):
        assert RouterAgent.resolve_agent("nonexistent") == "knowledge"


# ================================================================ #
#  Edge cases
# ================================================================ #

class TestEdgeCases:
    """Edge-case and stress tests."""

    def test_empty_query(self, agent):
        result = agent.classify_intent("")
        assert result["intent"] in INTENT_TYPES
        assert result["confidence"] <= 0.8

    def test_very_long_query(self, agent):
        long_query = "I want to improve my resume. " * 100
        result = agent.classify_intent(long_query)
        assert result["intent"] in ("resume_analysis", "resume_improvement")

    def test_with_user_context(self, agent):
        context = {
            "skills": ["Python", "FastAPI"],
            "target_roles": ["Backend Engineer"],
            "recent_activity": "resume_analysis",
        }
        result = agent.classify_intent(
            "Can you help me improve the bullet points?",
            user_context=context,
        )
        assert result["intent"] in ("resume_improvement", "resume_analysis", "career_advice")
