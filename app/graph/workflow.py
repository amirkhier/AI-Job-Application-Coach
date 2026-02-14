from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
import json
import logging
import re
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import JobCoachState, INTENT_TYPES
from app.agents.router import RouterAgent
from app.agents.memory import MemoryAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.resume import ResumeAgent
from app.agents.interview import InterviewAgent
from app.agents.job_search import JobSearchAgent

logger = logging.getLogger(__name__)


class JobCoachWorkflow:
    """Main workflow orchestrator for the AI Job Application Coach."""
    
    def __init__(
        self,
        router_agent: Optional[RouterAgent] = None,
        memory_agent: Optional[MemoryAgent] = None,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        resume_agent: Optional[ResumeAgent] = None,
        interview_agent: Optional[InterviewAgent] = None,
        job_search_agent: Optional[JobSearchAgent] = None,
        summary_llm: Optional[ChatOpenAI] = None,
    ):
        """Initialize the workflow with all agents and tools.
        
        Parameters
        ----------
        router_agent : RouterAgent, optional
            Injected router agent.  Created automatically if not supplied.
        memory_agent : MemoryAgent, optional
            Injected memory agent.  Created automatically if not supplied.
        knowledge_agent : KnowledgeAgent, optional
            Injected knowledge agent.  Created automatically if not supplied.
        resume_agent : ResumeAgent, optional
            Injected resume agent.  Created automatically if not supplied.
        interview_agent : InterviewAgent, optional
            Injected interview agent.  Created automatically if not supplied.
        job_search_agent : JobSearchAgent, optional
            Injected job search agent.  Created automatically if not supplied.
        summary_llm : ChatOpenAI, optional
            LLM used by the summary node.  Created automatically if not supplied.
        """
        self.router_agent = router_agent or RouterAgent()
        self.memory_agent = memory_agent or MemoryAgent()
        self.knowledge_agent = knowledge_agent or KnowledgeAgent()
        self.resume_agent = resume_agent or ResumeAgent()
        self.interview_agent = interview_agent or InterviewAgent()
        self.job_search_agent = job_search_agent or JobSearchAgent()
        self.summary_llm = summary_llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self._summary_prompt = self._build_summary_prompt()
        self.graph = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph state machine workflow."""
        
        # Create the state graph
        workflow = StateGraph(JobCoachState)
        
        # Add nodes for each agent
        workflow.add_node("router", self._router_agent)
        workflow.add_node("memory_load", self._memory_load_agent) 
        workflow.add_node("resume", self._resume_agent)
        workflow.add_node("interview", self._interview_agent)
        workflow.add_node("job_search", self._job_search_agent)
        workflow.add_node("knowledge", self._knowledge_agent)
        workflow.add_node("memory_save", self._memory_save_agent)
        workflow.add_node("summary", self._summary_agent)
        
        # Define entry point
        workflow.set_entry_point("memory_load")
        
        # Add edges from memory_load to router
        workflow.add_edge("memory_load", "router")
        
        # Add conditional routing from router to specialized agents
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "resume": "resume",
                "interview": "interview", 
                "job_search": "job_search",
                "knowledge": "knowledge",
                "summary": "summary"  # For unknown intents
            }
        )
        
        # All specialized agents go to summary
        workflow.add_edge("resume", "summary")
        workflow.add_edge("interview", "summary")
        workflow.add_edge("job_search", "summary")
        workflow.add_edge("knowledge", "summary")
        
        # Summary goes to memory save
        workflow.add_edge("summary", "memory_save")
        
        # Memory save goes to END
        workflow.add_edge("memory_save", END)
        
        # Compile the workflow
        self.graph = workflow.compile()
    
    def _router_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Router agent that classifies user intent and determines next agent."""
        start_time = time.time()
        
        try:
            user_query = state.get("user_query", "")
            user_context = state.get("user_profile")
            has_active_session = bool(state.get("interview_session_id"))

            classification = self.router_agent.classify_intent(
                user_query=user_query,
                user_context=user_context,
                has_active_session=has_active_session,
            )

            intent = classification["intent"]
            confidence = classification["confidence"]

            agents_used = state.get("agents_used", [])
            agents_used.append("router")

            logger.info(
                "Router: intent=%s  confidence=%.2f  method=%s  query='%s'",
                intent, confidence,
                classification.get("classification_method", "unknown"),
                user_query[:80],
            )
            
            return {
                "intent": intent,
                "confidence": confidence,
                "agents_used": agents_used,
                "debug_info": {
                    "router_processing_time": time.time() - start_time,
                    "classification_method": classification.get("classification_method"),
                    "classification_reasoning": classification.get("reasoning", ""),
                }
            }
            
        except Exception as e:
            logger.error("Router agent error: %s", e, exc_info=True)
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error_message": f"Router agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["router"]
            }
    
    def _route_to_agent(self, state: JobCoachState) -> str:
        """Determine which agent to route to based on classified intent."""
        intent = state.get("intent", "unknown")
        agent = RouterAgent.resolve_agent(intent)

        # If resolve_agent returns a valid graph node, use it;
        # otherwise fall through to summary for unknown intents.
        if agent in ("resume", "interview", "job_search", "knowledge"):
            return agent
        return "summary"  # Handle truly unmappable intents
    
    def _memory_load_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Load user profile and conversation history from memory.

        Calls :pymethod:`MemoryAgent.load_user_context` to retrieve the
        user's profile and recent conversations from MySQL.  On any failure
        the node returns empty defaults so the rest of the pipeline keeps
        running.
        """
        start_time = time.time()
        user_id = state.get("user_id", 1)

        try:
            context = self.memory_agent.load_user_context(
                user_id=user_id,
                interaction_type=state.get("intent", "general"),
                history_limit=5,
            )

            agents_used = state.get("agents_used", [])
            agents_used.append("memory_load")

            logger.info(
                "Memory load: user_id=%s  history_count=%s  processing=%.3fs",
                user_id,
                context.get("history_count", 0),
                time.time() - start_time,
            )

            return {
                "user_profile": context.get("profile", {}),
                "conversation_history": context.get("recent_conversations", []),
                "shared_context": context.get("context_summary", {}),
                "agents_used": agents_used,
                "debug_info": {
                    "memory_load_time": time.time() - start_time,
                    "history_count": context.get("history_count", 0),
                    "has_profile": bool(context.get("profile")),
                },
            }

        except Exception as e:
            logger.warning("Memory load failed for user %s: %s", user_id, e)
            return {
                "user_profile": {},
                "conversation_history": [],
                "shared_context": {},
                "agents_used": state.get("agents_used", []) + ["memory_load"],
                "debug_info": {
                    "memory_load_time": time.time() - start_time,
                    "memory_load_error": str(e),
                },
            }
    
    def _resume_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Resume analysis and improvement agent."""
        start_time = time.time()
        
        try:
            resume_text = state.get("resume_text") or ""
            job_description = state.get("job_description") or ""
            intent = state.get("intent") or "resume_analysis"
            
            if not resume_text:
                return {
                    "error_message": "No resume text provided for analysis",
                    "agents_used": state.get("agents_used", []) + ["resume"],
                }
            
            # Dispatch based on intent
            if intent == "resume_improvement":
                # Run analysis first, then improvements
                analysis = self.resume_agent.analyze_resume(resume_text, job_description)
                improvements = self.resume_agent.suggest_improvements(
                    resume_text, job_description, analysis
                )
                
                agents_used = state.get("agents_used", [])
                agents_used.append("resume")
                
                return {
                    "resume_analysis": analysis,
                    "resume_suggestions": improvements.get("priority_actions", []),
                    "agents_used": agents_used,
                    "debug_info": {
                        "resume_processing_time": time.time() - start_time,
                        "resume_length": len(resume_text),
                        "job_description_provided": bool(job_description),
                        "mode": "analysis_and_improvement",
                        "overall_score": analysis.get("overall_score"),
                        "improvement_bullets": len(improvements.get("improved_bullets", [])),
                    },
                }
            else:
                # Default: analysis only
                analysis = self.resume_agent.analyze_resume(resume_text, job_description)
                
                agents_used = state.get("agents_used", [])
                agents_used.append("resume")
                
                return {
                    "resume_analysis": analysis,
                    "agents_used": agents_used,
                    "debug_info": {
                        "resume_processing_time": time.time() - start_time,
                        "resume_length": len(resume_text),
                        "job_description_provided": bool(job_description),
                        "mode": "analysis_only",
                        "overall_score": analysis.get("overall_score"),
                    },
                }
            
        except Exception as e:
            logger.error("Resume agent error: %s", e, exc_info=True)
            return {
                "error_message": f"Resume agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["resume"],
            }
    
    def _interview_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Interview practice and evaluation agent.
        
        Dispatches based on intent:
        - interview_start / interview_practice → generate questions
        - interview_answer → evaluate the latest answer against the current question
        """
        start_time = time.time()
        
        try:
            role = state.get("interview_role") or "Software Engineer"
            level = state.get("interview_level") or "mid"
            intent = state.get("intent") or "interview_practice"
            
            if intent == "interview_answer":
                # ---- Evaluate an answer --------------------------------
                answers = state.get("interview_answers", [])
                questions = state.get("interview_questions", [])
                
                if not answers:
                    return {
                        "error_message": "No answer provided for evaluation",
                        "agents_used": state.get("agents_used", []) + ["interview"],
                    }
                
                # Latest answer to evaluate
                latest_answer = answers[-1]
                question_id = latest_answer.get("question_id", "")
                
                # Find the matching question
                current_question = next(
                    (q for q in questions if q.get("id") == question_id),
                    questions[0] if questions else {
                        "question": "General interview question",
                        "type": "behavioral",
                        "key_points": [],
                    },
                )
                
                evaluation = self.interview_agent.evaluate_answer(
                    question=current_question,
                    answer=latest_answer.get("answer", ""),
                    role=role,
                    level=level,
                )
                
                # Attach evaluation to the answer
                latest_answer["evaluation"] = evaluation
                
                agents_used = state.get("agents_used", [])
                agents_used.append("interview")
                
                return {
                    "interview_feedback": evaluation,
                    "interview_answers": answers,
                    "agents_used": agents_used,
                    "debug_info": {
                        "interview_processing_time": time.time() - start_time,
                        "mode": "evaluate_answer",
                        "question_id": question_id,
                        "overall_score": evaluation.get("overall_score"),
                    },
                }
            
            else:
                # ---- Generate questions (interview_start / interview_practice)
                questions = self.interview_agent.generate_questions(
                    role=role,
                    level=level,
                    count=5,
                )
                
                agents_used = state.get("agents_used", [])
                agents_used.append("interview")
                
                return {
                    "interview_questions": questions,
                    "agents_used": agents_used,
                    "debug_info": {
                        "interview_processing_time": time.time() - start_time,
                        "mode": "generate_questions",
                        "questions_generated": len(questions),
                        "role": role,
                        "level": level,
                    },
                }
            
        except Exception as e:
            logger.error("Interview agent error: %s", e, exc_info=True)
            return {
                "error_message": f"Interview agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["interview"],
            }
    
    def _job_search_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Job search and discovery agent."""
        start_time = time.time()
        
        try:
            query = state.get("job_search_query") or state.get("user_query", "Software Engineer")
            location = state.get("job_search_location") or "Remote"
            level = state.get("job_search_level") or "mid"
            user_profile = state.get("user_profile")
            
            # Use profile-enhanced search when a profile is available
            if user_profile:
                result = self.job_search_agent.search_jobs_with_matching(
                    query=query,
                    location=location,
                    experience_level=level,
                    remote_ok=True,
                    user_profile=user_profile,
                )
            else:
                result = self.job_search_agent.search_jobs(
                    query=query,
                    location=location,
                    experience_level=level,
                    remote_ok=True,
                )
            
            jobs = result.get("jobs", [])
            
            agents_used = state.get("agents_used", [])
            agents_used.append("job_search")
            
            return {
                "job_results": jobs,
                "agents_used": agents_used,
                "debug_info": {
                    "job_search_time": time.time() - start_time,
                    "results_found": len(jobs),
                    "search_query": query,
                    "search_location": location,
                    "search_level": level,
                    "profile_enhanced": bool(user_profile),
                    "location_info": result.get("location_info"),
                    "nearby_companies_found": len(result.get("nearby_companies", [])),
                },
            }
            
        except Exception as e:
            logger.error("Job search agent error: %s", e, exc_info=True)
            return {
                "error_message": f"Job search agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["job_search"],
            }
    
    def _knowledge_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Knowledge retrieval (RAG) agent for career advice."""
        start_time = time.time()
        
        try:
            query = state.get("knowledge_query") or state.get("user_query", "")
            
            if not query:
                return {
                    "error_message": "No query provided for knowledge retrieval",
                    "agents_used": state.get("agents_used", []) + ["knowledge"],
                }
            
            # Delegate to the real KnowledgeAgent
            result = self.knowledge_agent.answer_question(query)
            
            agents_used = state.get("agents_used", [])
            agents_used.append("knowledge")
            
            return {
                "knowledge_answer": result.get("answer", ""),
                "knowledge_sources": result.get("sources", []),
                "knowledge_context": f"RAG retrieval for: {query[:100]}",
                "agents_used": agents_used,
                "debug_info": {
                    "knowledge_retrieval_time": time.time() - start_time,
                    "query_length": len(query),
                    "sources_found": len(result.get("sources", [])),
                    "relevance_score": result.get("relevance_score", 0.0),
                    "context_chunks": result.get("context_chunks", 0),
                },
            }
            
        except Exception as e:
            logger.error("Knowledge agent error: %s", e, exc_info=True)
            return {
                "knowledge_answer": (
                    "I wasn't able to retrieve information from the knowledge base. "
                    "Please try rephrasing your question."
                ),
                "knowledge_sources": [],
                "agents_used": state.get("agents_used", []) + ["knowledge"],
                "debug_info": {
                    "knowledge_retrieval_time": time.time() - start_time,
                    "knowledge_error": str(e),
                },
            }
    
    def _memory_save_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Save conversation and update user profile in memory.

        Delegates to:
        1. ``MemoryAgent.save_conversation_with_analysis()`` — persists the
           user message + synthesised response with an LLM-generated summary.
        2. ``MemoryAgent.update_profile_from_conversation()`` — extracts new
           insights (skills, goals, preferences) and merges them into the
           user profile.

        Failures are logged in ``debug_info`` but never set ``error_message``
        so the pipeline can finish cleanly.
        """
        start_time = time.time()

        user_id = state.get("user_id", 1)
        session_id = state.get("session_id") or "unknown"
        user_query = state.get("user_query", "")
        response = state.get("response", "")
        intent = state.get("intent") or "unknown"

        # Determine which specialist agent handled this turn
        agents_used_so_far = state.get("agents_used", [])
        specialist = "general"
        for name in ("resume", "interview", "job_search", "knowledge"):
            if name in agents_used_so_far:
                specialist = name
                break

        save_result = None
        profile_result = None

        # ---- 1. Save conversation ------------------------------------
        try:
            save_result = self.memory_agent.save_conversation_with_analysis(
                user_id=user_id,
                session_id=session_id,
                user_message=user_query,
                agent_response=response,
                agent_type=specialist,
                intent=intent,
            )
        except Exception as exc:
            logger.warning("memory_save: conversation save failed: %s", exc)
            save_result = {"saved": False, "error": str(exc)}

        # ---- 2. Update profile from conversation ---------------------
        try:
            profile_result = self.memory_agent.update_profile_from_conversation(
                user_id=user_id,
                user_message=user_query,
                agent_response=response,
            )
        except Exception as exc:
            logger.warning("memory_save: profile update failed: %s", exc)
            profile_result = {"updated": False, "error": str(exc)}

        agents_used = list(agents_used_so_far)
        agents_used.append("memory_save")

        return {
            "profile_updates": profile_result.get("new_insights") if profile_result else None,
            "agents_used": agents_used,
            "debug_info": {
                "memory_save_time": time.time() - start_time,
                "user_id": user_id,
                "session_id": session_id,
                "conversation_saved": bool(save_result and save_result.get("saved")),
                "profile_updated": bool(profile_result and profile_result.get("updated")),
                "conversation_id": save_result.get("conversation_id") if save_result else None,
            },
        }
    
    # ------------------------------------------------------------------ #
    #  Summary prompt builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary_prompt() -> ChatPromptTemplate:
        """Build the prompt used by the LLM-powered summary node."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the response-synthesis layer of an AI career coach.\n\n"
                        "You receive:\n"
                        "1. The user's original question.\n"
                        "2. The detected intent.\n"
                        "3. Structured data produced by a specialised agent "
                        "(resume analyser, interview coach, job search engine, or knowledge base).\n"
                        "4. Optional user profile context.\n\n"
                        "Your job is to produce a polished, helpful, Markdown-formatted "
                        "response that the user will see directly.\n\n"
                        "RULES:\n"
                        "- Be conversational yet professional.\n"
                        "- Use Markdown headings, bullets, and bold for readability.\n"
                        "- Reference concrete data from the agent output (scores, names, sources, etc.).\n"
                        "- End with a brief, actionable suggestion or follow-up question.\n"
                        "- Do NOT reveal internal JSON, agent names, or system details.\n"
                        "- Keep the response focused — aim for 150-400 words.\n"
                        "- If the agent output is empty or an error occurred, "
                        "acknowledge it gracefully and suggest alternatives.\n"
                    ),
                ),
                (
                    "human",
                    (
                        "USER QUESTION: {user_query}\n\n"
                        "DETECTED INTENT: {intent}\n\n"
                        "AGENT OUTPUT:\n{agent_output}\n\n"
                        "USER PROFILE CONTEXT (may be empty):\n{user_context}\n\n"
                        "Generate the final user-facing response in Markdown."
                    ),
                ),
            ]
        )

    # ------------------------------------------------------------------ #
    #  Summary helpers
    # ------------------------------------------------------------------ #

    def _gather_agent_output(self, state: JobCoachState) -> str:
        """Collect the relevant agent output based on intent for the summary prompt."""
        intent = state.get("intent") or "unknown"
        parts: list[str] = []

        if intent in ("resume_analysis", "resume_improvement"):
            analysis = state.get("resume_analysis")
            if analysis:
                parts.append(json.dumps(analysis, indent=2, default=str))
            suggestions = state.get("resume_suggestions")
            if suggestions:
                parts.append(f"Priority actions: {json.dumps(suggestions)}")

        elif intent in ("interview_practice", "interview_start"):
            questions = state.get("interview_questions", [])
            if questions:
                parts.append(json.dumps(questions[:5], indent=2, default=str))
            parts.append(f"Role: {state.get('interview_role') or 'Software Engineer'}")
            parts.append(f"Level: {state.get('interview_level') or 'mid'}")

        elif intent == "interview_answer":
            feedback = state.get("interview_feedback")
            if feedback:
                parts.append(json.dumps(feedback, indent=2, default=str))
            # Include the question + answer for context
            answers = state.get("interview_answers", [])
            if answers:
                latest = answers[-1]
                parts.append(f"Question answered: {latest.get('question_id', '?')}")
                parts.append(f"Candidate answer: {latest.get('answer', '')[:500]}")

        elif intent == "job_search":
            jobs = state.get("job_results", [])
            if jobs:
                parts.append(json.dumps(jobs[:5], indent=2, default=str))

        elif intent in ("career_advice", "general_question", "application_tracking", "unknown"):
            answer = state.get("knowledge_answer", "")
            sources = state.get("knowledge_sources", [])
            if answer:
                parts.append(answer)
            if sources:
                parts.append(f"Sources: {', '.join(sources)}")

        error = state.get("error_message")
        if error:
            parts.append(f"ERROR: {error}")

        return "\n\n".join(parts) if parts else "No agent output available."

    @staticmethod
    def _gather_user_context(state: JobCoachState) -> str:
        """Build a concise user-context string from profile + history."""
        profile = state.get("user_profile")
        shared = state.get("shared_context")
        pieces: list[str] = []
        if profile:
            pieces.append(json.dumps(profile, indent=2, default=str))
        if shared:
            pieces.append(json.dumps(shared, indent=2, default=str))
        return "\n".join(pieces) if pieces else "No user profile available."

    def _template_fallback_response(self, state: JobCoachState) -> str:
        """Generate a simple template response when the LLM call fails."""
        intent = state.get("intent") or "unknown"
        error_message = state.get("error_message")

        if error_message:
            return f"I encountered an issue while processing your request: {error_message}"

        if intent in ("resume_analysis", "resume_improvement"):
            analysis = state.get("resume_analysis", {})
            if analysis:
                score = analysis.get("overall_score", 0)
                strengths = analysis.get("strengths", [])
                recs = analysis.get("recommendations", [])
                return (
                    f"## Resume Analysis\n\n"
                    f"**Score: {score}/10**\n\n"
                    f"**Strengths:** {', '.join(strengths[:3])}\n\n"
                    f"**Recommendations:** {', '.join(recs[:3])}"
                )
            return "I was unable to analyze your resume. Please ensure you've provided valid resume text."

        if intent in ("interview_practice", "interview_start"):
            questions = state.get("interview_questions", [])
            if questions:
                q = questions[0]
                return (
                    f"## Interview Practice\n\n"
                    f"**First question:** {q.get('question', 'Tell me about yourself.')}\n\n"
                    f"Take your time to answer."
                )
            return "I'm ready to start your interview practice. What role would you like to practice for?"

        if intent == "interview_answer":
            fb = state.get("interview_feedback", {})
            if fb:
                return (
                    f"## Answer Feedback\n\n"
                    f"**Score: {fb.get('overall_score', 'N/A')}/10**\n\n"
                    f"{fb.get('specific_feedback', 'Good effort — keep practising!')}"
                )
            return "I wasn't able to evaluate your answer. Please try again."

        if intent == "job_search":
            jobs = state.get("job_results", [])
            if jobs:
                lines = [f"- **{j.get('title')}** at {j.get('company')}" for j in jobs[:3]]
                return f"## Job Results\n\nFound {len(jobs)} opportunities:\n\n" + "\n".join(lines)
            return "No matching jobs found. Try broadening your search."

        if intent in ("career_advice", "general_question", "application_tracking", "unknown"):
            answer = state.get("knowledge_answer", "")
            if answer:
                sources = state.get("knowledge_sources", [])
                src = f"\n\n*Sources: {', '.join(sources)}*" if sources else ""
                return f"## Career Advice\n\n{answer}{src}"
            return "Could you be more specific about what you'd like guidance on?"

        return (
            "I can help with resume reviews, interview practice, job searches, "
            "and career advice. What would you like to work on?"
        )

    # ------------------------------------------------------------------ #
    #  Summary node
    # ------------------------------------------------------------------ #

    def _summary_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Synthesize a polished user-facing response from all agent outputs.

        Strategy:
        1. Gather structured agent output + user context.
        2. Call the summary LLM to produce a Markdown response.
        3. On LLM failure, fall back to the template renderer.
        """
        start_time = time.time()

        try:
            intent = state.get("intent") or "unknown"
            user_query = state.get("user_query", "")
            agent_output = self._gather_agent_output(state)
            user_context = self._gather_user_context(state)

            # --- LLM synthesis ---
            try:
                prompt_value = self._summary_prompt.invoke({
                    "user_query": user_query,
                    "intent": intent,
                    "agent_output": agent_output,
                    "user_context": user_context,
                })
                llm_result = self.summary_llm.invoke(prompt_value)
                response = llm_result.content.strip()
                synthesis_method = "llm"
            except Exception as llm_err:
                logger.warning(
                    "Summary LLM call failed, falling back to template: %s",
                    llm_err,
                )
                response = self._template_fallback_response(state)
                synthesis_method = "template_fallback"

            agents_used = state.get("agents_used", [])
            agents_used.append("summary")

            return {
                "response": response,
                "session_complete": True,
                "processing_time": time.time() - start_time,
                "agents_used": agents_used,
                "debug_info": {
                    "summary_time": time.time() - start_time,
                    "synthesis_method": synthesis_method,
                    "intent": intent,
                    "agent_output_length": len(agent_output),
                },
            }

        except Exception as e:
            logger.error("Summary agent error: %s", e, exc_info=True)
            return {
                "response": f"I encountered an error while generating your response: {str(e)}",
                "session_complete": True,
                "error_message": f"Summary agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["summary"],
            }
    
    def process_query(self, user_query: str, user_id: int = 1, session_id: str = None, **kwargs) -> Dict[str, Any]:
        """Process a user query through the complete workflow."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Initialize state
        initial_state: JobCoachState = {
            "user_query": user_query,
            "user_id": user_id, 
            "session_id": session_id,
            "intent": "",
            "confidence": 0.0,
            "resume_text": kwargs.get("resume_text"),
            "job_description": kwargs.get("job_description"),
            "resume_analysis": None,
            "resume_suggestions": None,
            "interview_role": kwargs.get("interview_role"),
            "interview_level": kwargs.get("interview_level"),
            "interview_questions": kwargs.get("interview_questions", []),
            "interview_answers": kwargs.get("interview_answers", []),
            "interview_feedback": None,
            "interview_session_id": kwargs.get("interview_session_id"),
            "job_search_query": kwargs.get("job_search_query"),
            "job_search_location": kwargs.get("job_search_location"),
            "job_search_level": kwargs.get("job_search_level"),
            "job_results": [],
            "knowledge_query": kwargs.get("knowledge_query", user_query),
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
            "debug_info": None
        }
        
        # Run the workflow
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            return {
                **initial_state,
                "response": f"I encountered a system error while processing your request: {str(e)}",
                "error_message": f"Workflow error: {str(e)}",
                "session_complete": True,
                "agents_used": ["error_handler"]
            }


# Global workflow instance
workflow = JobCoachWorkflow()


def get_workflow() -> JobCoachWorkflow:
    """Get the global workflow instance."""
    return workflow