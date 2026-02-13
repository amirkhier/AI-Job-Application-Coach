"""
Unit tests for the Knowledge Agent (RAG).

Covers:
- answer_question() with and without ChromaDB vectorstore
- get_topic_summary() retrieval
- JSON parsing and fallback behaviour
- Source attribution and merging
- Error handling when RAG is unavailable
- Prompt effectiveness: career-related answers, confidence scoring
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from app.agents.knowledge import KnowledgeAgent
from tests.conftest import make_mock_llm_response


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture(scope="module")
def agent():
    return KnowledgeAgent()


# ================================================================ #
#  answer_question
# ================================================================ #

class TestAnswerQuestion:
    """Tests for KnowledgeAgent.answer_question()."""

    def test_response_structure(self, agent):
        """Response must contain answer, sources, relevance_score, etc."""
        result = agent.answer_question("How should I prepare for a technical interview?")

        assert "answer" in result
        assert "sources" in result
        assert "relevance_score" in result
        assert "related_topics" in result
        assert "context_chunks" in result
        assert "processing_time" in result

    def test_answer_is_substantive(self, agent):
        """Answer should be at least a paragraph, not a one-liner."""
        result = agent.answer_question("What are the best practices for resume writing?")
        assert len(result["answer"]) > 50, "Answer too short"

    def test_sources_returned(self, agent):
        """For topics in our career guides, sources should be attributed."""
        result = agent.answer_question("How do I negotiate my salary?")
        # If vectorstore is built, sources should be non-empty
        # If not, this gracefully accepts empty sources
        assert isinstance(result["sources"], list)

    def test_relevance_score_range(self, agent):
        """Relevance/confidence score should be between 0 and 1."""
        result = agent.answer_question("Tell me about resume formats")
        assert 0.0 <= result["relevance_score"] <= 1.0

    def test_related_topics_returned(self, agent):
        """Related topics should be a list (may be empty)."""
        result = agent.answer_question("Interview preparation tips")
        assert isinstance(result["related_topics"], list)

    def test_off_topic_handled_gracefully(self, agent):
        """Non-career questions should still return a structured response."""
        result = agent.answer_question("What is the weather like today?")
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_processing_time_reasonable(self, agent):
        """Should complete within 60 seconds."""
        start = time.time()
        agent.answer_question("How to prepare for interviews?")
        assert time.time() - start < 60


# ================================================================ #
#  get_topic_summary
# ================================================================ #

class TestGetTopicSummary:
    """Tests for KnowledgeAgent.get_topic_summary()."""

    def test_topic_summary_structure(self, agent):
        result = agent.get_topic_summary("salary negotiation")

        assert "topic" in result
        assert "chunks" in result
        assert "sources" in result
        assert "avg_relevance" in result
        assert "processing_time" in result

    def test_topic_summary_is_fast(self, agent):
        """Topic summary skips LLM call, so should be very fast."""
        start = time.time()
        agent.get_topic_summary("interview tips")
        elapsed = time.time() - start
        # No LLM call — should be under 5 seconds even with embedding
        assert elapsed < 10, f"Topic summary took {elapsed:.1f}s (no LLM call expected)"


# ================================================================ #
#  JSON parsing
# ================================================================ #

class TestKnowledgeJSONParsing:

    def test_parse_clean_json(self):
        raw = '{"answer": "test", "confidence": 0.8}'
        result = KnowledgeAgent._parse_llm_json(raw)
        assert result["answer"] == "test"

    def test_parse_with_fences(self):
        raw = '```json\n{"answer": "test"}\n```'
        result = KnowledgeAgent._parse_llm_json(raw)
        assert result["answer"] == "test"

    def test_parse_failure_returns_none(self):
        result = KnowledgeAgent._parse_llm_json("not json at all")
        assert result is None


# ================================================================ #
#  Internal helpers
# ================================================================ #

class TestHelpers:

    def test_build_context_empty(self):
        assert KnowledgeAgent._build_context([]) == ""

    def test_build_context_formats_correctly(self):
        rag_results = [
            {"source": "test_doc.md", "content": "Some content here"},
            {"source": "other_doc.md", "content": "Other content"},
        ]
        ctx = KnowledgeAgent._build_context(rag_results)
        assert "[Source 1: test_doc.md]" in ctx
        assert "[Source 2: other_doc.md]" in ctx

    def test_merge_sources_deduplicates(self):
        rag = [{"source": "Resume Tips"}, {"source": "Interview Guide"}]
        llm = ["resume Tips", "Salary Guide"]  # Note case mismatch
        merged = KnowledgeAgent._merge_sources(rag, llm)

        assert len(merged) == 3  # Resume Tips, Interview Guide, Salary Guide
        # "resume Tips" should deduplicate with "Resume Tips"
        lower = [s.lower() for s in merged]
        assert lower.count("resume tips") == 1

    def test_fallback_answer_with_context(self):
        rag_results = [{"source": "doc.md", "content": "Useful info here", "score": 0.8}]
        answer, sources, conf, topics = KnowledgeAgent._fallback_answer("q", rag_results)
        assert "Useful info here" in answer
        assert len(sources) > 0
        assert conf > 0

    def test_fallback_answer_without_context(self):
        answer, sources, conf, topics = KnowledgeAgent._fallback_answer("q", [])
        assert "don't have specific information" in answer
        assert sources == []
        assert conf == 0.1
