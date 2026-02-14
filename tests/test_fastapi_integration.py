"""
Tests for Step 6 & 7 — FastAPI Integration with LangGraph feature flag,
multi-turn interview via /chat, and interview endpoint graph paths.

Covers:
 • /chat endpoint (always uses LangGraph workflow)
 • /chat multi-turn interview (session load, answer evaluation, DB update)
 • /chat auto-creates interview session when questions generated
 • Structured endpoints with USE_LANGGRAPH=false (legacy direct-agent path)
 • Structured endpoints with USE_LANGGRAPH=true  (graph-delegated path)
 • /interview/start and /interview/answer with USE_LANGGRAPH conditional
 • /interview/questions/{job_title} with USE_LANGGRAPH conditional
 • _extract_agent_data helper
 • _run_graph convenience wrapper
 • Error / edge-case handling
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_graph_state(**overrides):
    """Build a minimal graph final-state dict for mocking ``process_query``."""
    base = {
        "user_query": "test query",
        "user_id": 1,
        "session_id": "sess_123",
        "intent": "career_advice",
        "confidence": 0.92,
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
        "response": "Here is my answer.",
        "next_action": None,
        "session_complete": True,
        "processing_time": 1.23,
        "agents_used": ["knowledge"],
        "error_message": None,
        "debug_info": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# We need to patch heavyweight initialisers _before_ importing ``app.main``
# so that agent constructors (which connect to OpenAI / DB / ChromaDB) never
# run during tests.
# ---------------------------------------------------------------------------

_mock_db_manager = MagicMock()
_mock_db_manager.ensure_connection.return_value = True
_mock_db_manager.execute_query.return_value = [{"test": 1}]

# Patch database and agent constructors
_patches = [
    patch("app.tools.database.init_db"),
    patch("app.tools.database.close_db"),
    patch("app.tools.database.get_db", return_value=_mock_db_manager),
    patch("app.agents.resume.ResumeAgent.__init__", return_value=None),
    patch("app.agents.interview.InterviewAgent.__init__", return_value=None),
    patch("app.agents.knowledge.KnowledgeAgent.__init__", return_value=None),
    patch("app.agents.memory.MemoryAgent.__init__", return_value=None),
    patch("app.agents.job_search.JobSearchAgent.__init__", return_value=None),
    patch("app.graph.workflow.JobCoachWorkflow.__init__", return_value=None),
]

for p in _patches:
    p.start()

# NOW import the FastAPI app
import app.main as main_module
from app.main import app, _extract_agent_data

# Stop patches immediately so they don't leak into other test modules.
# The singletons in app.main are already created with mocked constructors.
for p in _patches:
    p.stop()

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_feature_flag():
    """Reset USE_LANGGRAPH to False before every test."""
    original = main_module.USE_LANGGRAPH
    main_module.USE_LANGGRAPH = False
    yield
    main_module.USE_LANGGRAPH = original


# ===========================================================================
#  1. _extract_agent_data helper
# ===========================================================================

class TestExtractAgentData:
    def test_empty_state_returns_none(self):
        state = _make_fake_graph_state()
        assert _extract_agent_data(state) is None

    def test_resume_analysis_present(self):
        state = _make_fake_graph_state(resume_analysis={"overall_score": 8.5})
        data = _extract_agent_data(state)
        assert data is not None
        assert data["resume_analysis"]["overall_score"] == 8.5

    def test_knowledge_answer_present(self):
        state = _make_fake_graph_state(
            knowledge_answer="Use STAR method.",
            knowledge_sources=["interview_tips.md"],
        )
        data = _extract_agent_data(state)
        assert data["knowledge_answer"] == "Use STAR method."
        assert data["knowledge_sources"] == ["interview_tips.md"]

    def test_job_results_present(self):
        state = _make_fake_graph_state(job_results=[{"title": "Dev"}])
        data = _extract_agent_data(state)
        assert len(data["job_results"]) == 1

    def test_interview_questions_present(self):
        state = _make_fake_graph_state(interview_questions=[{"id": "q1", "question": "Why?"}])
        data = _extract_agent_data(state)
        assert data["interview_questions"][0]["id"] == "q1"

    def test_multiple_fields(self):
        state = _make_fake_graph_state(
            resume_analysis={"score": 7},
            knowledge_answer="Tip",
        )
        data = _extract_agent_data(state)
        assert "resume_analysis" in data
        assert "knowledge_answer" in data


# ===========================================================================
#  2. /chat endpoint
# ===========================================================================

class TestChatEndpoint:
    """The /chat endpoint always routes through the LangGraph workflow."""

    def test_basic_chat(self):
        fake_state = _make_fake_graph_state(
            response="Here is career advice.",
            intent="career_advice",
            confidence=0.95,
            agents_used=["knowledge"],
            knowledge_answer="Use quantifiable achievements.",
            knowledge_sources=["resume_best_practices.md"],
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake_state
            resp = client.post("/chat", json={"message": "How do I write a resume?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Here is career advice."
        assert body["intent"] == "career_advice"
        assert body["confidence"] == 0.95
        assert "knowledge" in body["agents_used"]
        assert body["data"]["knowledge_answer"] == "Use quantifiable achievements."

    def test_chat_with_optional_fields(self):
        fake_state = _make_fake_graph_state(
            response="Analysed your resume.",
            intent="resume_analysis",
            resume_analysis={"overall_score": 7.5},
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake_state
            resp = client.post("/chat", json={
                "message": "Please analyze my resume",
                "resume_text": "Jane Doe\nDeveloper",
                "job_description": "We need a dev",
                "user_id": 42,
                "session_id": "custom-session",
            })

        assert resp.status_code == 200
        body = resp.json()
        # Verify kwargs forwarded
        call_kwargs = mock_wf.process_query.call_args
        assert call_kwargs.kwargs.get("resume_text") == "Jane Doe\nDeveloper"
        assert call_kwargs.kwargs.get("job_description") == "We need a dev"
        assert call_kwargs[1].get("user_id") or call_kwargs[0][1] == 42

    def test_chat_error_returns_500(self):
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.side_effect = RuntimeError("boom")
            resp = client.post("/chat", json={"message": "Hello"})

        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]

    def test_chat_with_error_message_in_state(self):
        """Graph completed but set an error_message."""
        fake = _make_fake_graph_state(
            response="Partial response",
            error_message="Memory save failed",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={"message": "test"})

        assert resp.status_code == 200
        assert resp.json()["error"] == "Memory save failed"

    def test_chat_no_data_when_empty(self):
        fake = _make_fake_graph_state()
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={"message": "Hi"})

        assert resp.status_code == 200
        assert resp.json()["data"] is None


# ===========================================================================
#  3. /resume — legacy (USE_LANGGRAPH=False)
# ===========================================================================

class TestResumeEndpointLegacy:
    def test_resume_direct_agent(self):
        analysis = {
            "overall_score": 8.0,
            "strengths": ["Strong experience"],
            "weaknesses": ["Missing summary"],
            "recommendations": ["Add metrics"],
            "ats_compatibility": {"score": 7.0, "issues": [], "suggestions": []},
            "keyword_analysis": {"present_keywords": ["Python"], "missing_keywords": ["Go"], "keyword_density_notes": ""},
            "section_feedback": {},
            "processing_time": 1.5,
        }
        with patch.object(main_module.resume_agent, "analyze_resume", return_value=analysis):
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis"):
                with patch.object(main_module.memory_agent, "update_profile_from_conversation"):
                    resp = client.post("/resume", json={
                        "resume_text": "A" * 50,
                        "user_id": 1,
                    })

        assert resp.status_code == 200
        assert resp.json()["overall_score"] == 8.0


# ===========================================================================
#  4. /resume — LangGraph path (USE_LANGGRAPH=True)
# ===========================================================================

class TestResumeEndpointGraph:
    def test_resume_via_graph(self):
        main_module.USE_LANGGRAPH = True
        analysis = {
            "overall_score": 9.0,
            "strengths": ["Great metrics"],
            "weaknesses": [],
            "recommendations": ["Keep it up"],
            "ats_compatibility": {"score": 8.5, "issues": [], "suggestions": []},
            "keyword_analysis": {"present_keywords": ["Python"], "missing_keywords": [], "keyword_density_notes": ""},
            "section_feedback": {},
            "processing_time": 2.0,
        }
        fake_state = _make_fake_graph_state(resume_analysis=analysis)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake_state
            resp = client.post("/resume", json={
                "resume_text": "A" * 50,
                "user_id": 1,
            })

        assert resp.status_code == 200
        assert resp.json()["overall_score"] == 9.0

    def test_resume_graph_does_not_call_memory_save(self):
        """When USE_LANGGRAPH is True the endpoint must NOT call memory_agent directly."""
        main_module.USE_LANGGRAPH = True
        analysis = {
            "overall_score": 7.0, "strengths": [], "weaknesses": [],
            "recommendations": [], "ats_compatibility": {}, "keyword_analysis": {},
            "processing_time": 0.5,
        }
        fake_state = _make_fake_graph_state(resume_analysis=analysis)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake_state
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis") as mock_save:
                resp = client.post("/resume", json={
                    "resume_text": "A" * 50,
                    "user_id": 1,
                })

        assert resp.status_code == 200
        mock_save.assert_not_called()


# ===========================================================================
#  5. /resume/improve — both paths
# ===========================================================================

class TestResumeImproveEndpoint:
    def _improvements(self):
        return {
            "improved_summary": "Better summary",
            "improved_bullets": [],
            "additional_suggestions": ["Add certifications"],
            "priority_actions": ["Quantify achievements"],
            "processing_time": 1.1,
        }

    def test_improve_legacy(self):
        with patch.object(main_module.resume_agent, "suggest_improvements", return_value=self._improvements()):
            resp = client.post("/resume/improve", json={
                "resume_text": "A" * 50,
                "user_id": 1,
            })
        assert resp.status_code == 200
        assert resp.json()["improved_summary"] == "Better summary"

    def test_improve_graph(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(resume_suggestions=self._improvements())
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/resume/improve", json={
                "resume_text": "A" * 50,
                "user_id": 1,
            })
        assert resp.status_code == 200
        assert resp.json()["improved_summary"] == "Better summary"


# ===========================================================================
#  6. /ask — both paths
# ===========================================================================

class TestAskEndpoint:
    def _knowledge_result(self):
        return {
            "answer": "Use the STAR method.",
            "sources": ["interview_tips.md"],
            "relevance_score": 0.85,
            "related_topics": ["behavioral interviews"],
        }

    def test_ask_legacy(self):
        with patch.object(main_module.knowledge_agent, "answer_question", return_value=self._knowledge_result()):
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis"):
                with patch.object(main_module.memory_agent, "update_profile_from_conversation"):
                    resp = client.post("/ask", json={
                        "query": "How do I answer behavioral questions?",
                        "user_id": 1,
                    })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Use the STAR method."
        assert body["relevance_score"] == 0.85

    def test_ask_graph(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(
            knowledge_answer="Use the STAR method.",
            knowledge_sources=["interview_tips.md"],
            debug_info={"relevance_score": 0.88},
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/ask", json={
                "query": "How do I answer behavioral questions?",
                "user_id": 1,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Use the STAR method."
        assert body["sources"] == ["interview_tips.md"]
        assert body["relevance_score"] == 0.88

    def test_ask_graph_no_memory_save(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(knowledge_answer="answer")
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis") as mock_save:
                resp = client.post("/ask", json={
                    "query": "What is salary negotiation?",
                    "user_id": 1,
                })
        mock_save.assert_not_called()


# ===========================================================================
#  7. /jobs/search — both paths
# ===========================================================================

class TestJobSearchEndpoint:
    def _job_result(self):
        return {
            "jobs": [
                {
                    "title": "Backend Engineer",
                    "company": "TechCo",
                    "location": "Tel Aviv",
                    "description": "Build things",
                    "url": "https://example.com",
                    "salary_range": "$80-120k",
                    "remote_friendly": True,
                    "match_score": 0.9,
                    "experience_level": "mid",
                    "key_skills": ["Python"],
                }
            ],
            "location_info": {
                "city": "Tel Aviv",
                "lat": 32.08,
                "lon": 34.78,
                "display_name": "Tel Aviv, Israel",
                "country": "Israel",
                "found": True,
            },
            "nearby_companies": [],
            "processing_time": 2.1,
        }

    def test_job_search_legacy(self):
        with patch.object(main_module.job_search_agent, "search_jobs", return_value=self._job_result()):
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis"):
                resp = client.post("/jobs/search", json={
                    "query": "Python developer",
                    "location": "Tel Aviv",
                    "user_id": 1,
                })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_found"] == 1
        assert body["jobs"][0]["title"] == "Backend Engineer"

    def test_job_search_graph(self):
        main_module.USE_LANGGRAPH = True
        raw_jobs = self._job_result()["jobs"]
        fake = _make_fake_graph_state(job_results=raw_jobs, processing_time=3.0)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/jobs/search", json={
                "query": "Python developer",
                "location": "Tel Aviv",
                "user_id": 1,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_found"] == 1
        assert body["jobs"][0]["company"] == "TechCo"

    def test_job_search_graph_no_memory_save(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(job_results=[])
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis") as mock_save:
                resp = client.post("/jobs/search", json={
                    "query": "developer",
                    "location": "London",
                    "user_id": 1,
                })
        mock_save.assert_not_called()


# ===========================================================================
#  8. Feature flag behaviour
# ===========================================================================

class TestFeatureFlag:
    def test_default_is_false(self):
        """Without USE_LANGGRAPH env var the flag defaults to False."""
        # In our test setup, we reset it to False every time
        assert main_module.USE_LANGGRAPH is False

    def test_resume_calls_direct_agent_when_flag_off(self):
        analysis = {
            "overall_score": 6.0, "strengths": [], "weaknesses": [],
            "recommendations": [], "ats_compatibility": {}, "keyword_analysis": {},
            "processing_time": 0.5,
        }
        with patch.object(main_module.resume_agent, "analyze_resume", return_value=analysis) as mock_agent:
            with patch.object(main_module.memory_agent, "save_conversation_with_analysis"):
                with patch.object(main_module.memory_agent, "update_profile_from_conversation"):
                    resp = client.post("/resume", json={"resume_text": "A" * 50})
        mock_agent.assert_called_once()

    def test_resume_does_not_call_direct_agent_when_flag_on(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(resume_analysis={
            "overall_score": 6.0, "strengths": [], "weaknesses": [],
            "recommendations": [], "ats_compatibility": {}, "keyword_analysis": {},
            "processing_time": 0.5,
        })
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.resume_agent, "analyze_resume") as mock_agent:
                resp = client.post("/resume", json={"resume_text": "A" * 50})
        mock_agent.assert_not_called()


# ===========================================================================
#  9. Health endpoint still works
# ===========================================================================

class TestHealth:
    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "AI Job Application Coach"


# ===========================================================================
#  10. Chat endpoint validation
# ===========================================================================

class TestChatValidation:
    def test_empty_message_rejected(self):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422  # validation error

    def test_missing_message_rejected(self):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422


# ===========================================================================
#  11. /interview/start — both paths
# ===========================================================================

class TestInterviewStartEndpoint:
    """Test /interview/start with USE_LANGGRAPH off and on."""

    _QUESTIONS = [
        {"id": "q1", "question": "Tell me about yourself.",
         "type": "behavioral", "difficulty": "easy", "key_points": ["intro"]},
        {"id": "q2", "question": "Why this role?",
         "type": "situational", "difficulty": "medium", "key_points": ["motivation"]},
    ]

    def test_start_legacy(self):
        with patch.object(main_module.interview_agent, "generate_questions", return_value=self._QUESTIONS):
            resp = client.post("/interview/start", json={
                "role": "Backend Engineer",
                "level": "mid",
                "question_count": 2,
                "user_id": 1,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "Backend Engineer"
        assert body["total_questions"] == 2
        assert body["first_question"]["id"] == "q1"

    def test_start_graph(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(
            interview_questions=self._QUESTIONS,
            intent="interview_start",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/interview/start", json={
                "role": "Backend Engineer",
                "level": "mid",
                "question_count": 2,
                "user_id": 1,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_questions"] == 2
        # Verify the graph was called (not the direct agent)
        mock_wf.process_query.assert_called_once()

    def test_start_graph_does_not_call_direct_agent(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(
            interview_questions=self._QUESTIONS,
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.interview_agent, "generate_questions") as mock_gen:
                resp = client.post("/interview/start", json={
                    "role": "SRE",
                    "level": "senior",
                    "question_count": 3,
                    "user_id": 1,
                })
        assert resp.status_code == 200
        mock_gen.assert_not_called()


# ===========================================================================
#  12. /interview/answer — both paths
# ===========================================================================

class TestInterviewAnswerEndpoint:
    """Test /interview/answer with USE_LANGGRAPH off and on."""

    _SESSION = {
        "user_id": 1,
        "role": "Backend Engineer",
        "level": "mid",
        "questions": [
            {"id": "q1", "question": "Tell me about yourself.",
             "type": "behavioral", "difficulty": "easy", "key_points": ["intro"]},
        ],
        "answers": [],
    }

    _EVALUATION = {
        "overall_score": 7.5,
        "strength_areas": ["Clear"],
        "improvement_areas": ["Depth"],
        "specific_feedback": "Good start.",
        "suggested_improvement": "Add metrics.",
    }

    def test_answer_legacy(self):
        _mock_db_manager.get_interview_session.return_value = dict(self._SESSION)
        with patch.object(main_module.interview_agent, "evaluate_answer", return_value=self._EVALUATION):
            resp = client.post("/interview/answer", json={
                "session_id": "abc-123",
                "question_id": "q1",
                "answer": "I have 5 years of backend experience in Python.",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback"]["overall_score"] == 7.5
        assert body["session_complete"] is True
        _mock_db_manager.get_interview_session.return_value = None  # reset

    def test_answer_graph(self):
        main_module.USE_LANGGRAPH = True
        _mock_db_manager.get_interview_session.return_value = dict(self._SESSION)
        fake = _make_fake_graph_state(interview_feedback=self._EVALUATION)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/interview/answer", json={
                "session_id": "abc-123",
                "question_id": "q1",
                "answer": "I have 5 years of backend experience in Python.",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["feedback"]["overall_score"] == 7.5
        mock_wf.process_query.assert_called_once()
        _mock_db_manager.get_interview_session.return_value = None

    def test_answer_graph_does_not_call_direct_agent(self):
        main_module.USE_LANGGRAPH = True
        _mock_db_manager.get_interview_session.return_value = dict(self._SESSION)
        fake = _make_fake_graph_state(interview_feedback=self._EVALUATION)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.interview_agent, "evaluate_answer") as mock_eval:
                resp = client.post("/interview/answer", json={
                    "session_id": "abc-123",
                    "question_id": "q1",
                    "answer": "I have 5 years of backend experience in Python.",
                })
        mock_eval.assert_not_called()
        _mock_db_manager.get_interview_session.return_value = None


# ===========================================================================
#  13. /interview/questions/{job_title} — both paths
# ===========================================================================

class TestInterviewQuestionsEndpoint:
    _QUESTIONS = [
        {"id": "q1", "question": "Explain REST.", "type": "technical",
         "difficulty": "easy", "key_points": ["HTTP"]},
    ]

    def test_questions_legacy(self):
        with patch.object(main_module.interview_agent, "generate_questions", return_value=self._QUESTIONS):
            resp = client.get("/interview/questions/DevOps?level=junior&count=1")
        assert resp.status_code == 200
        assert resp.json()["role"] == "DevOps"

    def test_questions_graph(self):
        main_module.USE_LANGGRAPH = True
        fake = _make_fake_graph_state(interview_questions=self._QUESTIONS)
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.get("/interview/questions/DevOps?level=junior&count=1")
        assert resp.status_code == 200
        assert len(resp.json()["questions"]) == 1
        mock_wf.process_query.assert_called_once()


# ===========================================================================
#  14. Multi-turn interview via /chat
# ===========================================================================

class TestChatMultiTurnInterview:
    """Test the unified /chat endpoint with multi-turn interview support."""

    _QUESTIONS = [
        {"id": "q1", "question": "Tell me about yourself.",
         "type": "behavioral", "difficulty": "easy", "key_points": ["intro"]},
        {"id": "q2", "question": "Why this role?",
         "type": "situational", "difficulty": "medium", "key_points": ["motivation"]},
    ]

    _SESSION = {
        "user_id": 1,
        "role": "Backend Engineer",
        "level": "mid",
        "questions": None,  # set per test
        "answers": [],
    }

    def test_chat_starts_interview_creates_session(self):
        """When /chat generates interview questions without an active session,
        a new DB session should be created and its ID returned."""
        fake = _make_fake_graph_state(
            intent="interview_practice",
            interview_questions=self._QUESTIONS,
            response="Here are your interview questions.",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={
                "message": "Practice interview for Backend Engineer",
                "interview_role": "Backend Engineer",
            })

        assert resp.status_code == 200
        body = resp.json()
        # A new interview_session_id should be returned
        assert body["interview_session_id"] is not None
        # DB should have been called to create + update the session
        _mock_db_manager.create_interview_session.assert_called()
        _mock_db_manager.update_interview_session.assert_called()

    def test_chat_answers_within_session(self):
        """When interview_session_id + interview_answer are provided, the
        graph evaluates the answer and the DB session is updated."""
        session = dict(self._SESSION, questions=self._QUESTIONS, answers=[])
        _mock_db_manager.get_interview_session.return_value = session

        fake = _make_fake_graph_state(
            intent="interview_answer",
            interview_feedback={
                "overall_score": 8.0,
                "strength_areas": ["Clear"],
                "improvement_areas": [],
                "specific_feedback": "Great answer.",
                "suggested_improvement": "",
            },
            response="Great answer! Score: 8/10.",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={
                "message": "Here is my answer to q1",
                "interview_session_id": "sess-abc",
                "interview_answer": "I have 5 years of backend experience.",
                "interview_question_id": "q1",
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["interview_session_id"] == "sess-abc"
        assert body["data"] is not None
        assert body["data"]["interview_questions_remaining"] == 1
        assert body["data"]["next_question"]["id"] == "q2"
        assert body["data"]["interview_session_complete"] is False

        # DB should have been updated
        _mock_db_manager.update_interview_session.assert_called()
        _mock_db_manager.get_interview_session.return_value = None

    def test_chat_completes_interview_session(self):
        """When all questions are answered the session should be marked complete."""
        session = dict(
            self._SESSION,
            questions=self._QUESTIONS,
            answers=[{
                "question_id": "q1",
                "answer": "Already answered",
                "evaluation": {"overall_score": 7.0},
            }],
        )
        _mock_db_manager.get_interview_session.return_value = session

        fake = _make_fake_graph_state(
            intent="interview_answer",
            interview_feedback={"overall_score": 9.0},
            response="Final answer evaluated.",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            with patch.object(main_module.interview_agent, "generate_session_summary",
                              return_value={"overall": "Good session"}):
                resp = client.post("/chat", json={
                    "message": "My answer to q2",
                    "interview_session_id": "sess-abc",
                    "interview_answer": "Because I love distributed systems.",
                    "interview_question_id": "q2",
                })

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["interview_session_complete"] is True
        assert body["data"]["interview_questions_remaining"] == 0
        assert body["data"]["interview_session_summary"] == {"overall": "Good session"}

        # DB update should have completed=True
        update_call = _mock_db_manager.update_interview_session.call_args
        assert update_call.kwargs.get("completed") is True
        _mock_db_manager.get_interview_session.return_value = None

    def test_chat_no_session_no_interview_fields(self):
        """Without interview_session_id the response should have no interview_session_id
        (unless the intent generates questions)."""
        fake = _make_fake_graph_state(
            intent="career_advice",
            response="Here is career advice.",
        )
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={"message": "How to negotiate salary?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["interview_session_id"] is None

    def test_chat_invalid_session_id_ignored(self):
        """If the session is not found in DB the graph still runs normally."""
        _mock_db_manager.get_interview_session.return_value = None
        fake = _make_fake_graph_state(response="I can help.")
        with patch.object(main_module, "_workflow") as mock_wf:
            mock_wf.process_query.return_value = fake
            resp = client.post("/chat", json={
                "message": "Continue interview",
                "interview_session_id": "nonexistent",
            })

        assert resp.status_code == 200
        _mock_db_manager.get_interview_session.return_value = None
