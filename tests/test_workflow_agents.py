"""
Unit tests for the workflow specialized agent nodes (Step 3).

Validates that the workflow's knowledge, resume, job_search, and interview
nodes correctly delegate to their real agent classes and map results to
the JobCoachState schema.

Covers per agent:
- Happy path: agent called with correct args, result mapped to state
- Error fallback: agent raises, node degrades gracefully
- State field mapping: agent output keys → state keys
- agents_used list correctly appended
- debug_info populated with timing and metadata
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from app.agents.router import RouterAgent
from app.agents.memory import MemoryAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.resume import ResumeAgent
from app.agents.interview import InterviewAgent
from app.agents.job_search import JobSearchAgent
from app.graph.workflow import JobCoachWorkflow


# ================================================================ #
#  Shared helpers & fixtures
# ================================================================ #

def _make_state(**overrides):
    """Build a minimal JobCoachState dict for testing."""
    base = {
        "user_query": "Tell me about salary negotiation",
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
        "knowledge_query": "Tell me about salary negotiation",
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


@pytest.fixture
def mock_router():
    agent = MagicMock(spec=RouterAgent)
    agent.classify_intent.return_value = {
        "intent": "career_advice",
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
    return agent


# ================================================================ #
#  KNOWLEDGE AGENT NODE
# ================================================================ #

class TestKnowledgeAgentNode:
    """Tests for _knowledge_agent workflow node."""

    @pytest.fixture
    def mock_knowledge(self):
        agent = MagicMock(spec=KnowledgeAgent)
        agent.answer_question.return_value = {
            "answer": "Research market rates before negotiating.",
            "sources": ["salary_negotiation.md", "career_guides"],
            "relevance_score": 0.85,
            "related_topics": ["compensation", "benefits"],
            "context_chunks": 3,
            "processing_time": 0.5,
        }
        return agent

    @pytest.fixture
    def workflow(self, mock_router, mock_memory, mock_knowledge):
        return JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            knowledge_agent=mock_knowledge,
        )

    def test_calls_answer_question(self, workflow, mock_knowledge):
        state = _make_state(knowledge_query="How to negotiate salary?")
        workflow._knowledge_agent(state)
        mock_knowledge.answer_question.assert_called_once_with("How to negotiate salary?")

    def test_falls_back_to_user_query(self, workflow, mock_knowledge):
        state = _make_state(knowledge_query=None, user_query="tips for salary talk")
        workflow._knowledge_agent(state)
        mock_knowledge.answer_question.assert_called_once_with("tips for salary talk")

    def test_answer_mapped_to_state(self, workflow):
        state = _make_state()
        result = workflow._knowledge_agent(state)
        assert result["knowledge_answer"] == "Research market rates before negotiating."

    def test_sources_mapped(self, workflow):
        state = _make_state()
        result = workflow._knowledge_agent(state)
        assert "salary_negotiation.md" in result["knowledge_sources"]

    def test_agents_used(self, workflow):
        state = _make_state(agents_used=["memory_load", "router"])
        result = workflow._knowledge_agent(state)
        assert result["agents_used"] == ["memory_load", "router", "knowledge"]

    def test_debug_info(self, workflow):
        state = _make_state()
        result = workflow._knowledge_agent(state)
        assert "knowledge_retrieval_time" in result["debug_info"]
        assert result["debug_info"]["sources_found"] == 2
        assert result["debug_info"]["relevance_score"] == 0.85
        assert result["debug_info"]["context_chunks"] == 3

    def test_no_error_message_on_success(self, workflow):
        state = _make_state()
        result = workflow._knowledge_agent(state)
        assert "error_message" not in result

    def test_empty_query_returns_error(self, workflow):
        state = _make_state(knowledge_query="", user_query="")
        result = workflow._knowledge_agent(state)
        assert "error_message" in result
        assert "No query" in result["error_message"]

    def test_agent_failure_graceful(self, mock_router, mock_memory):
        failing = MagicMock(spec=KnowledgeAgent)
        failing.answer_question.side_effect = RuntimeError("ChromaDB down")
        wf = JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            knowledge_agent=failing,
        )
        state = _make_state()
        result = wf._knowledge_agent(state)

        # Should NOT raise, should return a fallback answer
        assert "knowledge_answer" in result
        assert result["knowledge_sources"] == []
        assert "knowledge" in result["agents_used"]
        assert "ChromaDB down" in result["debug_info"]["knowledge_error"]
        # Critically, error_message is NOT set — the summary node would
        # treat the fallback answer as a normal response
        assert "error_message" not in result


# ================================================================ #
#  RESUME AGENT NODE
# ================================================================ #

class TestResumeAgentNode:
    """Tests for _resume_agent workflow node."""

    MOCK_ANALYSIS = {
        "overall_score": 7.8,
        "strengths": ["Strong experience", "Good skills"],
        "weaknesses": ["Missing metrics"],
        "recommendations": ["Add numbers"],
        "ats_compatibility": {"score": 8.0, "issues": [], "suggestions": []},
        "keyword_analysis": {"present_keywords": ["Python"], "missing_keywords": ["Kafka"]},
        "section_feedback": {"experience": "Good"},
        "processing_time": 0.9,
    }

    MOCK_IMPROVEMENTS = {
        "improved_summary": "Experienced backend engineer...",
        "improved_bullets": [
            {"original": "Built APIs", "improved": "Architected REST APIs serving 50K users", "reasoning": "Quantified"}
        ],
        "additional_suggestions": ["Add certifications"],
        "priority_actions": ["Quantify achievements", "Tailor to role"],
        "processing_time": 1.1,
    }

    @pytest.fixture
    def mock_resume(self):
        agent = MagicMock(spec=ResumeAgent)
        agent.analyze_resume.return_value = self.MOCK_ANALYSIS.copy()
        agent.suggest_improvements.return_value = self.MOCK_IMPROVEMENTS.copy()
        return agent

    @pytest.fixture
    def workflow(self, mock_router, mock_memory, mock_resume):
        return JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            resume_agent=mock_resume,
        )

    def test_analysis_mode_calls_analyze(self, workflow, mock_resume):
        state = _make_state(resume_text="My resume...", intent="resume_analysis")
        workflow._resume_agent(state)
        mock_resume.analyze_resume.assert_called_once_with("My resume...", "")
        mock_resume.suggest_improvements.assert_not_called()

    def test_improvement_mode_calls_both(self, workflow, mock_resume):
        state = _make_state(
            resume_text="My resume...",
            job_description="Senior Python dev",
            intent="resume_improvement",
        )
        workflow._resume_agent(state)
        mock_resume.analyze_resume.assert_called_once_with("My resume...", "Senior Python dev")
        mock_resume.suggest_improvements.assert_called_once()

    def test_analysis_result_mapped(self, workflow):
        state = _make_state(resume_text="My resume...", intent="resume_analysis")
        result = workflow._resume_agent(state)
        assert result["resume_analysis"]["overall_score"] == 7.8

    def test_improvement_populates_suggestions(self, workflow):
        state = _make_state(resume_text="My resume...", intent="resume_improvement")
        result = workflow._resume_agent(state)
        assert result["resume_suggestions"] == ["Quantify achievements", "Tailor to role"]

    def test_no_resume_returns_error(self, workflow):
        state = _make_state(resume_text="", intent="resume_analysis")
        result = workflow._resume_agent(state)
        assert "error_message" in result
        assert "No resume" in result["error_message"]

    def test_agents_used(self, workflow):
        state = _make_state(resume_text="x", agents_used=["router"])
        result = workflow._resume_agent(state)
        assert result["agents_used"] == ["router", "resume"]

    def test_debug_info_analysis_mode(self, workflow):
        state = _make_state(resume_text="My resume text here", intent="resume_analysis")
        result = workflow._resume_agent(state)
        assert result["debug_info"]["mode"] == "analysis_only"
        assert result["debug_info"]["resume_length"] == len("My resume text here")
        assert result["debug_info"]["overall_score"] == 7.8

    def test_debug_info_improvement_mode(self, workflow):
        state = _make_state(resume_text="txt", intent="resume_improvement")
        result = workflow._resume_agent(state)
        assert result["debug_info"]["mode"] == "analysis_and_improvement"
        assert result["debug_info"]["improvement_bullets"] == 1

    def test_agent_failure(self, mock_router, mock_memory):
        failing = MagicMock(spec=ResumeAgent)
        failing.analyze_resume.side_effect = Exception("LLM timeout")
        wf = JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            resume_agent=failing,
        )
        state = _make_state(resume_text="Some resume")
        result = wf._resume_agent(state)
        assert "error_message" in result
        assert "LLM timeout" in result["error_message"]
        assert "resume" in result["agents_used"]

    def test_job_description_forwarded(self, workflow, mock_resume):
        state = _make_state(
            resume_text="resume",
            job_description="JD here",
            intent="resume_analysis",
        )
        workflow._resume_agent(state)
        mock_resume.analyze_resume.assert_called_once_with("resume", "JD here")


# ================================================================ #
#  JOB SEARCH AGENT NODE
# ================================================================ #

class TestJobSearchAgentNode:
    """Tests for _job_search_agent workflow node."""

    MOCK_SEARCH_RESULT = {
        "jobs": [
            {
                "title": "Senior Python Engineer",
                "company": "TechCo",
                "location": "Tel Aviv",
                "description": "Build APIs",
                "salary_range": "₪35K-₪50K/mo",
                "remote_friendly": True,
                "match_score": 0.9,
            }
        ],
        "total_found": 1,
        "search_query": "Python Engineer",
        "location": "Tel Aviv",
        "location_info": {"lat": 32.08, "lon": 34.78, "display_name": "Tel Aviv"},
        "nearby_companies": [{"name": "Wix"}],
        "processing_time": 1.2,
    }

    @pytest.fixture
    def mock_job_search(self):
        agent = MagicMock(spec=JobSearchAgent)
        agent.search_jobs.return_value = self.MOCK_SEARCH_RESULT.copy()
        agent.search_jobs_with_matching.return_value = self.MOCK_SEARCH_RESULT.copy()
        return agent

    @pytest.fixture
    def workflow(self, mock_router, mock_memory, mock_job_search):
        return JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            job_search_agent=mock_job_search,
        )

    def test_calls_search_jobs(self, workflow, mock_job_search):
        state = _make_state(
            job_search_query="Python Engineer",
            job_search_location="Tel Aviv",
            job_search_level="senior",
        )
        workflow._job_search_agent(state)
        mock_job_search.search_jobs.assert_called_once_with(
            query="Python Engineer",
            location="Tel Aviv",
            experience_level="senior",
            remote_ok=True,
        )

    def test_uses_profile_enhanced_search(self, workflow, mock_job_search):
        profile = {"skills": {"technical": ["Python"]}, "experience_level": "senior"}
        state = _make_state(
            job_search_query="Dev",
            job_search_location="Berlin",
            user_profile=profile,
        )
        workflow._job_search_agent(state)
        mock_job_search.search_jobs_with_matching.assert_called_once()
        mock_job_search.search_jobs.assert_not_called()

    def test_jobs_mapped_to_state(self, workflow):
        state = _make_state(job_search_query="Python Engineer")
        result = workflow._job_search_agent(state)
        assert len(result["job_results"]) == 1
        assert result["job_results"][0]["company"] == "TechCo"

    def test_agents_used(self, workflow):
        state = _make_state(agents_used=["memory_load"])
        result = workflow._job_search_agent(state)
        assert result["agents_used"] == ["memory_load", "job_search"]

    def test_debug_info(self, workflow):
        state = _make_state(
            job_search_query="Dev",
            job_search_location="TLV",
            job_search_level="mid",
        )
        result = workflow._job_search_agent(state)
        assert result["debug_info"]["results_found"] == 1
        assert result["debug_info"]["search_query"] == "Dev"
        assert result["debug_info"]["search_location"] == "TLV"
        assert result["debug_info"]["profile_enhanced"] is False

    def test_defaults_when_no_query(self, workflow, mock_job_search):
        """Falls back to user_query when job_search_query is missing."""
        state = _make_state(
            job_search_query=None,
            user_query="Find me a backend job",
        )
        workflow._job_search_agent(state)
        args = mock_job_search.search_jobs.call_args
        assert args.kwargs["query"] == "Find me a backend job"

    def test_agent_failure(self, mock_router, mock_memory):
        failing = MagicMock(spec=JobSearchAgent)
        failing.search_jobs.side_effect = Exception("Nominatim timeout")
        wf = JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            job_search_agent=failing,
        )
        state = _make_state(job_search_query="Dev")
        result = wf._job_search_agent(state)
        assert "error_message" in result
        assert "Nominatim timeout" in result["error_message"]
        assert "job_search" in result["agents_used"]


# ================================================================ #
#  INTERVIEW AGENT NODE
# ================================================================ #

class TestInterviewAgentNode:
    """Tests for _interview_agent workflow node."""

    MOCK_QUESTIONS = [
        {
            "id": "q1",
            "question": "Tell me about yourself.",
            "type": "behavioral",
            "difficulty": "easy",
            "key_points": ["Background", "Motivation"],
        },
        {
            "id": "q2",
            "question": "Describe a technical challenge.",
            "type": "technical",
            "difficulty": "medium",
            "key_points": ["Problem solving", "Technical depth"],
        },
    ]

    MOCK_EVALUATION = {
        "overall_score": 7.5,
        "dimension_scores": {
            "relevance": 8.0,
            "depth": 7.0,
            "structure": 7.5,
            "communication": 7.5,
        },
        "strength_areas": ["Good use of examples"],
        "improvement_areas": ["Could add more metrics"],
        "specific_feedback": "Solid answer with room for improvement.",
        "suggested_improvement": "Use STAR method more explicitly.",
        "processing_time": 0.8,
    }

    @pytest.fixture
    def mock_interview(self):
        agent = MagicMock(spec=InterviewAgent)
        agent.generate_questions.return_value = [q.copy() for q in self.MOCK_QUESTIONS]
        agent.evaluate_answer.return_value = self.MOCK_EVALUATION.copy()
        return agent

    @pytest.fixture
    def workflow(self, mock_router, mock_memory, mock_interview):
        return JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            interview_agent=mock_interview,
        )

    # ---- Question generation tests ----

    def test_generate_questions_on_start(self, workflow, mock_interview):
        state = _make_state(
            intent="interview_start",
            interview_role="Backend Engineer",
            interview_level="senior",
        )
        workflow._interview_agent(state)
        mock_interview.generate_questions.assert_called_once_with(
            role="Backend Engineer", level="senior", count=5,
        )

    def test_generate_questions_on_practice(self, workflow, mock_interview):
        state = _make_state(intent="interview_practice")
        workflow._interview_agent(state)
        mock_interview.generate_questions.assert_called_once()

    def test_questions_mapped_to_state(self, workflow):
        state = _make_state(intent="interview_start")
        result = workflow._interview_agent(state)
        assert len(result["interview_questions"]) == 2
        assert result["interview_questions"][0]["id"] == "q1"

    def test_generate_debug_info(self, workflow):
        state = _make_state(intent="interview_start", interview_role="PM")
        result = workflow._interview_agent(state)
        assert result["debug_info"]["mode"] == "generate_questions"
        assert result["debug_info"]["questions_generated"] == 2
        assert result["debug_info"]["role"] == "PM"

    # ---- Answer evaluation tests ----

    def test_evaluate_answer(self, workflow, mock_interview):
        questions = [q.copy() for q in self.MOCK_QUESTIONS]
        answers = [{"question_id": "q1", "answer": "I have 7 years of experience..."}]
        state = _make_state(
            intent="interview_answer",
            interview_questions=questions,
            interview_answers=answers,
            interview_role="Backend Engineer",
            interview_level="senior",
        )
        workflow._interview_agent(state)
        mock_interview.evaluate_answer.assert_called_once_with(
            question=questions[0],
            answer="I have 7 years of experience...",
            role="Backend Engineer",
            level="senior",
        )

    def test_evaluation_mapped_to_feedback(self, workflow):
        questions = [q.copy() for q in self.MOCK_QUESTIONS]
        answers = [{"question_id": "q1", "answer": "My answer"}]
        state = _make_state(
            intent="interview_answer",
            interview_questions=questions,
            interview_answers=answers,
        )
        result = workflow._interview_agent(state)
        assert result["interview_feedback"]["overall_score"] == 7.5

    def test_evaluation_attached_to_answer(self, workflow):
        questions = [q.copy() for q in self.MOCK_QUESTIONS]
        answers = [{"question_id": "q1", "answer": "My answer"}]
        state = _make_state(
            intent="interview_answer",
            interview_questions=questions,
            interview_answers=answers,
        )
        result = workflow._interview_agent(state)
        # The evaluation should be attached to the answer dict
        assert result["interview_answers"][-1]["evaluation"]["overall_score"] == 7.5

    def test_evaluate_debug_info(self, workflow):
        questions = [q.copy() for q in self.MOCK_QUESTIONS]
        answers = [{"question_id": "q2", "answer": "I solved..."}]
        state = _make_state(
            intent="interview_answer",
            interview_questions=questions,
            interview_answers=answers,
        )
        result = workflow._interview_agent(state)
        assert result["debug_info"]["mode"] == "evaluate_answer"
        assert result["debug_info"]["question_id"] == "q2"
        assert result["debug_info"]["overall_score"] == 7.5

    def test_no_answer_returns_error(self, workflow):
        state = _make_state(intent="interview_answer", interview_answers=[])
        result = workflow._interview_agent(state)
        assert "error_message" in result
        assert "No answer" in result["error_message"]

    # ---- Shared tests ----

    def test_agents_used_generate(self, workflow):
        state = _make_state(intent="interview_start", agents_used=["router"])
        result = workflow._interview_agent(state)
        assert result["agents_used"] == ["router", "interview"]

    def test_agents_used_evaluate(self, workflow):
        state = _make_state(
            intent="interview_answer",
            interview_questions=[self.MOCK_QUESTIONS[0].copy()],
            interview_answers=[{"question_id": "q1", "answer": "x"}],
            agents_used=["router"],
        )
        result = workflow._interview_agent(state)
        assert result["agents_used"] == ["router", "interview"]

    def test_default_role_and_level(self, workflow, mock_interview):
        state = _make_state(intent="interview_start")
        workflow._interview_agent(state)
        mock_interview.generate_questions.assert_called_once_with(
            role="Software Engineer", level="mid", count=5,
        )

    def test_agent_failure(self, mock_router, mock_memory):
        failing = MagicMock(spec=InterviewAgent)
        failing.generate_questions.side_effect = Exception("LLM rate limit")
        wf = JobCoachWorkflow(
            router_agent=mock_router,
            memory_agent=mock_memory,
            interview_agent=failing,
        )
        state = _make_state(intent="interview_start")
        result = wf._interview_agent(state)
        assert "error_message" in result
        assert "LLM rate limit" in result["error_message"]
        assert "interview" in result["agents_used"]


# ================================================================ #
#  INTEGRATION: full pipeline with mocked agents
# ================================================================ #

class TestFullPipelineWithMockedAgents:
    """End-to-end graph run with all agents mocked — no LLM calls."""

    @pytest.fixture
    def all_mocks(self):
        router = MagicMock(spec=RouterAgent)
        memory = MagicMock(spec=MemoryAgent)
        knowledge = MagicMock(spec=KnowledgeAgent)
        resume = MagicMock(spec=ResumeAgent)
        interview = MagicMock(spec=InterviewAgent)
        job_search = MagicMock(spec=JobSearchAgent)

        router.classify_intent.return_value = {
            "intent": "career_advice",
            "confidence": 0.95,
            "reasoning": "User asked career question",
            "classification_method": "keyword",
            "processing_time": 0.01,
        }
        memory.load_user_context.return_value = {
            "user_id": 1,
            "profile": {"experience_level": "mid"},
            "preferences": {},
            "recent_conversations": [],
            "context_summary": {"user_background": "Test user"},
            "history_count": 0,
            "processing_time": 0.01,
        }
        knowledge.answer_question.return_value = {
            "answer": "Here is your career advice.",
            "sources": ["career_guides"],
            "relevance_score": 0.8,
            "related_topics": [],
            "context_chunks": 2,
            "processing_time": 0.3,
        }
        return router, memory, knowledge, resume, interview, job_search

    def test_career_advice_pipeline(self, all_mocks):
        router, memory, knowledge, resume, interview, job_search = all_mocks
        wf = JobCoachWorkflow(
            router_agent=router,
            memory_agent=memory,
            knowledge_agent=knowledge,
            resume_agent=resume,
            interview_agent=interview,
            job_search_agent=job_search,
        )
        result = wf.process_query("How do I negotiate salary?", user_id=1)

        assert result["session_complete"] is True
        assert "knowledge" in result["agents_used"]
        assert "router" in result["agents_used"]
        assert "memory_load" in result["agents_used"]
        # Knowledge answer should appear in state
        assert result["knowledge_answer"] == "Here is your career advice."
        # Resume/interview/job_search should NOT have been called
        resume.analyze_resume.assert_not_called()
        interview.generate_questions.assert_not_called()
        job_search.search_jobs.assert_not_called()

    def test_resume_analysis_pipeline(self, all_mocks):
        router, memory, knowledge, resume, interview, job_search = all_mocks
        router.classify_intent.return_value["intent"] = "resume_analysis"
        resume.analyze_resume.return_value = {
            "overall_score": 8.0,
            "strengths": ["Good"],
            "weaknesses": ["Minor issues"],
            "recommendations": ["Polish"],
            "ats_compatibility": {"score": 7.5},
            "keyword_analysis": {},
            "section_feedback": {},
            "processing_time": 1.0,
        }
        wf = JobCoachWorkflow(
            router_agent=router,
            memory_agent=memory,
            knowledge_agent=knowledge,
            resume_agent=resume,
            interview_agent=interview,
            job_search_agent=job_search,
        )
        result = wf.process_query(
            "Review my resume",
            user_id=1,
            resume_text="My resume content",
        )

        assert result["session_complete"] is True
        assert "resume" in result["agents_used"]
        resume.analyze_resume.assert_called_once()
        knowledge.answer_question.assert_not_called()

    def test_interview_pipeline(self, all_mocks):
        router, memory, knowledge, resume, interview, job_search = all_mocks
        router.classify_intent.return_value["intent"] = "interview_practice"
        interview.generate_questions.return_value = [
            {"id": "q1", "question": "Tell me about yourself.", "type": "behavioral",
             "difficulty": "easy", "key_points": ["background"]},
        ]
        wf = JobCoachWorkflow(
            router_agent=router,
            memory_agent=memory,
            knowledge_agent=knowledge,
            resume_agent=resume,
            interview_agent=interview,
            job_search_agent=job_search,
        )
        result = wf.process_query("Start a mock interview", user_id=1)

        assert result["session_complete"] is True
        assert "interview" in result["agents_used"]
        interview.generate_questions.assert_called_once()

    def test_job_search_pipeline(self, all_mocks):
        router, memory, knowledge, resume, interview, job_search = all_mocks
        router.classify_intent.return_value["intent"] = "job_search"
        job_search.search_jobs.return_value = {
            "jobs": [{"title": "Dev", "company": "Co"}],
            "total_found": 1,
            "search_query": "Dev",
            "location": "Remote",
            "location_info": {},
            "nearby_companies": [],
            "processing_time": 0.5,
        }
        wf = JobCoachWorkflow(
            router_agent=router,
            memory_agent=memory,
            knowledge_agent=knowledge,
            resume_agent=resume,
            interview_agent=interview,
            job_search_agent=job_search,
        )
        result = wf.process_query("Find me jobs in Berlin", user_id=1)

        assert result["session_complete"] is True
        assert "job_search" in result["agents_used"]
        # Memory loads a profile, so the profile-enhanced path is used
        job_search.search_jobs_with_matching.assert_called_once()
