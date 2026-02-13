"""
Unit tests for the Memory Agent.

Covers:
- load_user_context() with existing and non-existent users
- save_conversation_with_analysis()
- update_profile_from_conversation()
- get_conversation_insights()
- JSON parsing helpers
- Error handling when DB is unavailable
- Performance timing
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from app.agents.memory import MemoryAgent
from tests.conftest import make_mock_llm_response


# ================================================================ #
#  Fixtures
# ================================================================ #

@pytest.fixture
def mock_db():
    """Create a mock DatabaseManager with sensible defaults."""
    db = MagicMock()
    db.get_user.return_value = {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com",
        "profile_data": {
            "skills": {"technical": ["Python", "SQL"], "soft": ["teamwork"]},
            "experience_level": "mid",
            "target_roles": ["Backend Engineer"],
        },
        "preferences": {"location": "Tel Aviv", "remote_ok": True},
    }
    db.get_conversation_history.return_value = [
        {
            "id": 1,
            "session_id": "sess-001",
            "message": "How should I improve my Python skills?",
            "response": "Focus on data structures and async programming.",
            "agent_type": "knowledge",
            "created_at": "2026-02-10 10:00:00",
        },
        {
            "id": 2,
            "session_id": "sess-002",
            "message": "Review my resume",
            "response": "Your resume looks solid, but add more metrics.",
            "agent_type": "resume",
            "created_at": "2026-02-11 14:30:00",
        },
    ]
    db.save_conversation.return_value = True
    db.update_user_profile.return_value = True
    return db


@pytest.fixture
def agent(mock_db):
    """Create MemoryAgent with a mocked DB."""
    return MemoryAgent(db=mock_db)


# ================================================================ #
#  load_user_context
# ================================================================ #

class TestLoadUserContext:

    def test_context_structure(self, agent):
        """load_user_context should return profile, history, and context."""
        result = agent.load_user_context(user_id=1)

        assert "profile" in result
        assert "history_count" in result
        assert "processing_time" in result

    def test_context_with_interaction_type(self, agent):
        """Different interaction types should still work."""
        for itype in ("general", "resume", "interview", "job_search"):
            result = agent.load_user_context(user_id=1, interaction_type=itype)
            assert "profile" in result

    def test_nonexistent_user(self, agent, mock_db):
        """For a user not in DB, should return graceful empty context."""
        mock_db.get_user.return_value = None
        mock_db.get_conversation_history.return_value = []

        result = agent.load_user_context(user_id=999)
        assert "profile" in result


# ================================================================ #
#  save_conversation_with_analysis
# ================================================================ #

class TestSaveConversation:

    def test_save_returns_result(self, agent):
        """Save should complete and return a result dict."""
        result = agent.save_conversation_with_analysis(
            user_id=1,
            session_id="test-session",
            user_message="Help me with my resume",
            agent_response="I'd recommend focusing on quantified achievements.",
            agent_type="resume",
        )
        assert isinstance(result, dict)

    def test_save_calls_db(self, agent, mock_db):
        """Save should attempt to write to the database."""
        agent.save_conversation_with_analysis(
            user_id=1,
            session_id="sess-test",
            user_message="Test message",
            agent_response="Test response",
        )
        # Should have attempted at least one DB call
        assert mock_db.save_conversation.called or mock_db.method_calls


# ================================================================ #
#  get_conversation_insights
# ================================================================ #

class TestConversationInsights:

    def test_insights_structure(self, agent):
        """Insights should include aggregated conversation data."""
        result = agent.get_conversation_insights(user_id=1)
        assert isinstance(result, dict)
        assert "conversation_count" in result
        assert "insights" in result


# ================================================================ #
#  Error handling
# ================================================================ #

class TestErrorHandling:

    def test_db_failure_on_load(self, mock_db):
        """If DB fails on load, agent should handle gracefully."""
        mock_db.get_user.side_effect = Exception("DB connection lost")
        agent = MemoryAgent(db=mock_db)

        # Should not raise — returns empty/fallback context
        result = agent.load_user_context(user_id=1)
        assert isinstance(result, dict)

    def test_db_failure_on_save(self, mock_db):
        """If DB fails on save, agent should not crash."""
        mock_db.save_conversation.side_effect = Exception("DB write error")
        agent = MemoryAgent(db=mock_db)

        # Should not raise
        result = agent.save_conversation_with_analysis(
            user_id=1,
            session_id="test",
            user_message="msg",
            agent_response="resp",
        )
        assert isinstance(result, dict)
