"""
Unit tests for the workflow memory-save node (Step 5).

Validates that _memory_save_agent:
- Calls MemoryAgent.save_conversation_with_analysis() with correct args
- Calls MemoryAgent.update_profile_from_conversation() with correct args
- Maps results to state (profile_updates, debug_info)
- Degrades gracefully when either call fails (no error_message set)
- Appends 'memory_save' to agents_used
- Infers specialist agent from agents_used list
"""

import pytest
from unittest.mock import MagicMock

from app.agents.router import RouterAgent
from app.agents.memory import MemoryAgent
from app.graph.workflow import JobCoachWorkflow


# ================================================================ #
#  Helpers & fixtures
# ================================================================ #

def _make_state(**overrides):
    base = {
        "user_query": "Review my resume please",
        "user_id": 1,
        "session_id": "sess-42",
        "intent": "resume_analysis",
        "confidence": 0.9,
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
        "knowledge_query": None,
        "knowledge_context": None,
        "knowledge_sources": [],
        "knowledge_answer": None,
        "user_profile": None,
        "conversation_history": [],
        "profile_updates": None,
        "agent_messages": [],
        "shared_context": None,
        "response": "Your resume scored 7.5/10.",
        "next_action": None,
        "session_complete": True,
        "processing_time": 0.0,
        "agents_used": ["memory_load", "router", "resume", "summary"],
        "error_message": None,
        "debug_info": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def mock_router():
    agent = MagicMock(spec=RouterAgent)
    agent.classify_intent.return_value = {
        "intent": "resume_analysis",
        "confidence": 0.9,
        "reasoning": "test",
        "classification_method": "keyword",
        "processing_time": 0.01,
    }
    return agent


@pytest.fixture
def mock_memory():
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
    agent.save_conversation_with_analysis.return_value = {
        "conversation_id": 101,
        "summary": {"key_topics": ["resume"], "sentiment": "positive"},
        "saved": True,
        "processing_time": 0.15,
    }
    agent.update_profile_from_conversation.return_value = {
        "updated": True,
        "new_insights": {"skills_mentioned": ["Python", "FastAPI"]},
        "full_profile": {"skills": {"technical": ["Python", "FastAPI"]}},
        "processing_time": 0.12,
    }
    return agent


@pytest.fixture
def workflow(mock_router, mock_memory):
    return JobCoachWorkflow(
        router_agent=mock_router,
        memory_agent=mock_memory,
    )


# ================================================================ #
#  Happy-path tests
# ================================================================ #

class TestMemorySaveHappyPath:
    """Verify the node delegates correctly and maps state."""

    def test_calls_save_conversation(self, workflow, mock_memory):
        state = _make_state()
        workflow._memory_save_agent(state)

        mock_memory.save_conversation_with_analysis.assert_called_once_with(
            user_id=1,
            session_id="sess-42",
            user_message="Review my resume please",
            agent_response="Your resume scored 7.5/10.",
            agent_type="resume",
            intent="resume_analysis",
        )

    def test_calls_update_profile(self, workflow, mock_memory):
        state = _make_state()
        workflow._memory_save_agent(state)

        mock_memory.update_profile_from_conversation.assert_called_once_with(
            user_id=1,
            user_message="Review my resume please",
            agent_response="Your resume scored 7.5/10.",
        )

    def test_conversation_saved_in_debug_info(self, workflow):
        state = _make_state()
        result = workflow._memory_save_agent(state)
        assert result["debug_info"]["conversation_saved"] is True
        assert result["debug_info"]["conversation_id"] == 101

    def test_profile_updated_in_debug_info(self, workflow):
        state = _make_state()
        result = workflow._memory_save_agent(state)
        assert result["debug_info"]["profile_updated"] is True

    def test_profile_updates_populated(self, workflow):
        state = _make_state()
        result = workflow._memory_save_agent(state)
        assert result["profile_updates"] == {"skills_mentioned": ["Python", "FastAPI"]}

    def test_agents_used_appended(self, workflow):
        state = _make_state(agents_used=["memory_load", "router", "resume", "summary"])
        result = workflow._memory_save_agent(state)
        assert result["agents_used"][-1] == "memory_save"
        assert "resume" in result["agents_used"]

    def test_no_error_message_set(self, workflow):
        state = _make_state()
        result = workflow._memory_save_agent(state)
        assert "error_message" not in result

    def test_timing_in_debug_info(self, workflow):
        state = _make_state()
        result = workflow._memory_save_agent(state)
        assert result["debug_info"]["memory_save_time"] >= 0


# ================================================================ #
#  Specialist inference tests
# ================================================================ #

class TestSpecialistInference:
    """The node should detect which agent handled the query."""

    def test_resume_specialist(self, workflow, mock_memory):
        state = _make_state(agents_used=["memory_load", "router", "resume", "summary"])
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_type"] == "resume"

    def test_interview_specialist(self, workflow, mock_memory):
        state = _make_state(agents_used=["memory_load", "router", "interview", "summary"])
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_type"] == "interview"

    def test_knowledge_specialist(self, workflow, mock_memory):
        state = _make_state(
            agents_used=["memory_load", "router", "knowledge", "summary"],
            intent="career_advice",
        )
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_type"] == "knowledge"

    def test_job_search_specialist(self, workflow, mock_memory):
        state = _make_state(agents_used=["memory_load", "router", "job_search", "summary"])
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_type"] == "job_search"

    def test_no_specialist_defaults_to_general(self, workflow, mock_memory):
        state = _make_state(agents_used=["memory_load", "router", "summary"])
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_type"] == "general"


# ================================================================ #
#  Graceful failure tests
# ================================================================ #

class TestMemorySaveFailure:
    """Memory save failures must NOT crash the pipeline."""

    def test_save_failure_no_error_message(self, mock_router):
        mem = MagicMock(spec=MemoryAgent)
        mem.load_user_context.return_value = {
            "user_id": 1, "profile": {}, "preferences": {},
            "recent_conversations": [], "context_summary": {},
            "history_count": 0, "processing_time": 0.01,
        }
        mem.save_conversation_with_analysis.side_effect = Exception("DB down")
        mem.update_profile_from_conversation.return_value = {
            "updated": False, "new_insights": {}, "full_profile": {},
            "processing_time": 0.01,
        }
        wf = JobCoachWorkflow(router_agent=mock_router, memory_agent=mem)

        state = _make_state()
        result = wf._memory_save_agent(state)

        assert "error_message" not in result
        assert result["debug_info"]["conversation_saved"] is False
        assert "memory_save" in result["agents_used"]

    def test_profile_update_failure_no_error_message(self, mock_router):
        mem = MagicMock(spec=MemoryAgent)
        mem.load_user_context.return_value = {
            "user_id": 1, "profile": {}, "preferences": {},
            "recent_conversations": [], "context_summary": {},
            "history_count": 0, "processing_time": 0.01,
        }
        mem.save_conversation_with_analysis.return_value = {
            "conversation_id": 99, "summary": None, "saved": True,
            "processing_time": 0.1,
        }
        mem.update_profile_from_conversation.side_effect = RuntimeError("LLM timeout")
        wf = JobCoachWorkflow(router_agent=mock_router, memory_agent=mem)

        state = _make_state()
        result = wf._memory_save_agent(state)

        assert "error_message" not in result
        assert result["debug_info"]["conversation_saved"] is True
        assert result["debug_info"]["profile_updated"] is False

    def test_both_fail_still_completes(self, mock_router):
        mem = MagicMock(spec=MemoryAgent)
        mem.load_user_context.return_value = {
            "user_id": 1, "profile": {}, "preferences": {},
            "recent_conversations": [], "context_summary": {},
            "history_count": 0, "processing_time": 0.01,
        }
        mem.save_conversation_with_analysis.side_effect = Exception("DB gone")
        mem.update_profile_from_conversation.side_effect = Exception("Also gone")
        wf = JobCoachWorkflow(router_agent=mock_router, memory_agent=mem)

        state = _make_state()
        result = wf._memory_save_agent(state)

        assert "error_message" not in result
        assert result["debug_info"]["conversation_saved"] is False
        assert result["debug_info"]["profile_updated"] is False
        assert "memory_save" in result["agents_used"]

    def test_pipeline_completes_after_save_failure(self, mock_router):
        """Full graph run: memory_save failure should not prevent completion."""
        mem = MagicMock(spec=MemoryAgent)
        mem.load_user_context.return_value = {
            "user_id": 1, "profile": {}, "preferences": {},
            "recent_conversations": [], "context_summary": {},
            "history_count": 0, "processing_time": 0.01,
        }
        mem.save_conversation_with_analysis.side_effect = Exception("DB crash")
        mem.update_profile_from_conversation.side_effect = Exception("DB crash")
        wf = JobCoachWorkflow(router_agent=mock_router, memory_agent=mem)

        result = wf.process_query("Give me career advice", user_id=1)

        assert result["session_complete"] is True
        assert "memory_save" in result["agents_used"]
        assert "summary" in result["agents_used"]


# ================================================================ #
#  Edge cases
# ================================================================ #

class TestMemorySaveEdgeCases:
    """Edge-case and data-shape validation."""

    def test_empty_response(self, workflow, mock_memory):
        state = _make_state(response="")
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["agent_response"] == ""

    def test_none_session_id(self, workflow, mock_memory):
        state = _make_state(session_id=None)
        workflow._memory_save_agent(state)
        call_kwargs = mock_memory.save_conversation_with_analysis.call_args.kwargs
        assert call_kwargs["session_id"] == "unknown"

    def test_preserves_existing_agents_used(self, workflow):
        state = _make_state(agents_used=["a", "b", "c"])
        result = workflow._memory_save_agent(state)
        assert result["agents_used"] == ["a", "b", "c", "memory_save"]

    def test_no_profile_updates_when_not_updated(self, mock_router):
        mem = MagicMock(spec=MemoryAgent)
        mem.load_user_context.return_value = {
            "user_id": 1, "profile": {}, "preferences": {},
            "recent_conversations": [], "context_summary": {},
            "history_count": 0, "processing_time": 0.01,
        }
        mem.save_conversation_with_analysis.return_value = {
            "conversation_id": 50, "summary": None, "saved": True,
            "processing_time": 0.05,
        }
        mem.update_profile_from_conversation.return_value = {
            "updated": False, "new_insights": {}, "full_profile": {},
            "processing_time": 0.01,
        }
        wf = JobCoachWorkflow(router_agent=mock_router, memory_agent=mem)

        state = _make_state()
        result = wf._memory_save_agent(state)
        assert result["profile_updates"] == {}
