"""
Unit tests for the workflow _memory_load_agent node (Step 2).

Validates that the workflow's memory_load node correctly delegates to
the real MemoryAgent.load_user_context() and maps its output to the
JobCoachState schema.

Covers:
- Happy path: profile + history loaded and mapped to state
- Graceful fallback: DB/agent failure returns empty defaults (no crash)
- State field mapping: profile → user_profile, conversations → conversation_history
- shared_context population from context_summary
- agents_used list correctly appended
- debug_info populated with timing and metadata
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.agents.router import RouterAgent
from app.agents.memory import MemoryAgent
from app.graph.workflow import JobCoachWorkflow


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture
def mock_memory_agent():
    """MemoryAgent with a stubbed load_user_context()."""
    agent = MagicMock(spec=MemoryAgent)
    agent.load_user_context.return_value = {
        "user_id": 1,
        "profile": {
            "skills": {"technical": ["Python", "FastAPI"], "soft": ["teamwork"]},
            "experience_level": "senior",
            "target_roles": ["Backend Engineer"],
        },
        "preferences": {"location": "Tel Aviv", "remote_ok": True},
        "recent_conversations": [
            {
                "id": 10,
                "session_id": "sess-100",
                "message": "Review my resume",
                "response": "Your resume looks solid.",
                "agent_used": "resume",
                "created_at": "2026-02-10T10:00:00",
            }
        ],
        "context_summary": {
            "user_background": "Senior developer with 7 years experience",
            "relevant_history": "Recent resume review",
            "preferences": "Prefers Tel Aviv, open to remote",
            "context_notes": "Focus area: backend roles",
        },
        "history_count": 1,
        "processing_time": 0.123,
    }
    return agent


@pytest.fixture
def mock_router_agent():
    """RouterAgent with a stubbed classify_intent()."""
    agent = MagicMock(spec=RouterAgent)
    agent.classify_intent.return_value = {
        "intent": "career_advice",
        "confidence": 0.9,
        "reasoning": "test",
        "classification_method": "llm",
        "processing_time": 0.05,
    }
    return agent


@pytest.fixture
def workflow(mock_router_agent, mock_memory_agent):
    """Workflow with mocked router and memory agents."""
    return JobCoachWorkflow(
        router_agent=mock_router_agent,
        memory_agent=mock_memory_agent,
    )


def _make_state(**overrides):
    """Build a minimal JobCoachState dict for testing."""
    base = {
        "user_query": "How do I negotiate my salary?",
        "user_id": 1,
        "session_id": "test-session",
        "intent": "",
        "confidence": 0.0,
        "resume_text": None,
        "job_description": None,
        "resume_analysis": None,
        "resume_suggestions": None,
        "interview_role": None,
        "interview_level": None,
        "interview_questions": [],
        "interview_answers": [],
        "interview_feedback": None,
        "interview_session_id": None,
        "job_search_query": None,
        "job_search_location": None,
        "job_search_level": None,
        "job_results": [],
        "knowledge_query": "How do I negotiate my salary?",
        "knowledge_context": None,
        "knowledge_sources": [],
        "knowledge_answer": None,
        "user_profile": None,
        "conversation_history": [],
        "profile_updates": None,
        "agent_messages": [],
        "shared_context": None,
        "response": "",
        "next_action": None,
        "session_complete": False,
        "processing_time": 0.0,
        "agents_used": [],
        "error_message": None,
        "debug_info": None,
    }
    base.update(overrides)
    return base


# ================================================================ #
#  Happy-path tests
# ================================================================ #

class TestMemoryLoadHappyPath:
    """Verify the node delegates to MemoryAgent and maps state correctly."""

    def test_calls_load_user_context(self, workflow, mock_memory_agent):
        """The node must call MemoryAgent.load_user_context with the user_id."""
        state = _make_state(user_id=42)
        workflow._memory_load_agent(state)

        mock_memory_agent.load_user_context.assert_called_once_with(
            user_id=42,
            interaction_type="",  # intent is empty at this point
            history_limit=5,
        )

    def test_user_profile_populated(self, workflow):
        """user_profile should come from the 'profile' key of the context."""
        state = _make_state()
        result = workflow._memory_load_agent(state)

        assert result["user_profile"]["experience_level"] == "senior"
        assert "Python" in result["user_profile"]["skills"]["technical"]

    def test_conversation_history_populated(self, workflow):
        """conversation_history should come from 'recent_conversations'."""
        state = _make_state()
        result = workflow._memory_load_agent(state)

        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["agent_used"] == "resume"

    def test_shared_context_populated(self, workflow):
        """shared_context should come from 'context_summary'."""
        state = _make_state()
        result = workflow._memory_load_agent(state)

        assert "user_background" in result["shared_context"]
        assert "Senior developer" in result["shared_context"]["user_background"]

    def test_agents_used_appended(self, workflow):
        """memory_load should be appended to agents_used."""
        state = _make_state(agents_used=["some_prior"])
        result = workflow._memory_load_agent(state)

        assert "memory_load" in result["agents_used"]
        assert "some_prior" in result["agents_used"]

    def test_debug_info_present(self, workflow):
        """debug_info should contain timing and metadata."""
        state = _make_state()
        result = workflow._memory_load_agent(state)

        assert "memory_load_time" in result["debug_info"]
        assert result["debug_info"]["history_count"] == 1
        assert result["debug_info"]["has_profile"] is True

    def test_no_error_message_set(self, workflow):
        """On success, error_message should NOT be set in the return dict."""
        state = _make_state()
        result = workflow._memory_load_agent(state)

        assert "error_message" not in result


# ================================================================ #
#  Graceful fallback tests
# ================================================================ #

class TestMemoryLoadFallback:
    """Verify the node degrades gracefully when MemoryAgent fails."""

    def test_db_failure_returns_empty_defaults(self, mock_router_agent):
        """If load_user_context raises, the node returns empty dicts/lists."""
        failing_agent = MagicMock(spec=MemoryAgent)
        failing_agent.load_user_context.side_effect = Exception("DB connection refused")

        wf = JobCoachWorkflow(
            router_agent=mock_router_agent,
            memory_agent=failing_agent,
        )
        state = _make_state()
        result = wf._memory_load_agent(state)

        assert result["user_profile"] == {}
        assert result["conversation_history"] == []
        assert result["shared_context"] == {}

    def test_db_failure_still_appends_agents_used(self, mock_router_agent):
        """Even on failure, memory_load should appear in agents_used."""
        failing_agent = MagicMock(spec=MemoryAgent)
        failing_agent.load_user_context.side_effect = RuntimeError("timeout")

        wf = JobCoachWorkflow(
            router_agent=mock_router_agent,
            memory_agent=failing_agent,
        )
        state = _make_state()
        result = wf._memory_load_agent(state)

        assert "memory_load" in result["agents_used"]

    def test_db_failure_records_error_in_debug_info(self, mock_router_agent):
        """Failure error message should appear in debug_info, not error_message."""
        failing_agent = MagicMock(spec=MemoryAgent)
        failing_agent.load_user_context.side_effect = ConnectionError("MySQL gone")

        wf = JobCoachWorkflow(
            router_agent=mock_router_agent,
            memory_agent=failing_agent,
        )
        state = _make_state()
        result = wf._memory_load_agent(state)

        assert "memory_load_error" in result["debug_info"]
        assert "MySQL gone" in result["debug_info"]["memory_load_error"]
        # Crucially, error_message is NOT set — it would cause the summary
        # node to render an error response instead of forwarding to agents.
        assert "error_message" not in result

    def test_db_failure_does_not_block_pipeline(self, mock_router_agent):
        """After a memory failure the graph should still reach the summary node."""
        failing_agent = MagicMock(spec=MemoryAgent)
        failing_agent.load_user_context.side_effect = Exception("DB down")

        wf = JobCoachWorkflow(
            router_agent=mock_router_agent,
            memory_agent=failing_agent,
        )

        # Run the full graph — it should NOT raise
        result = wf.process_query("Give me career advice", user_id=99)

        assert result["session_complete"] is True
        assert "memory_load" in result["agents_used"]
        assert "router" in result["agents_used"]


# ================================================================ #
#  Edge-case tests
# ================================================================ #

class TestMemoryLoadEdgeCases:
    """Edge cases and data-shape validation."""

    def test_empty_profile_from_agent(self, mock_router_agent):
        """If MemoryAgent returns an empty profile, state gets {}."""
        agent = MagicMock(spec=MemoryAgent)
        agent.load_user_context.return_value = {
            "user_id": 1,
            "profile": {},
            "preferences": {},
            "recent_conversations": [],
            "context_summary": {},
            "history_count": 0,
            "processing_time": 0.01,
        }

        wf = JobCoachWorkflow(
            router_agent=mock_router_agent,
            memory_agent=agent,
        )
        state = _make_state()
        result = wf._memory_load_agent(state)

        assert result["user_profile"] == {}
        assert result["conversation_history"] == []
        assert result["debug_info"]["has_profile"] is False
        assert result["debug_info"]["history_count"] == 0

    def test_preserves_existing_agents_used(self, workflow):
        """Pre-existing agents_used entries must survive."""
        state = _make_state(agents_used=["alpha", "beta"])
        result = workflow._memory_load_agent(state)

        assert result["agents_used"] == ["alpha", "beta", "memory_load"]

    def test_different_user_ids(self, workflow, mock_memory_agent):
        """Should forward whichever user_id is in the state."""
        for uid in [1, 42, 999]:
            workflow._memory_load_agent(_make_state(user_id=uid))

        calls = mock_memory_agent.load_user_context.call_args_list
        assert [c.kwargs["user_id"] for c in calls] == [1, 42, 999]
