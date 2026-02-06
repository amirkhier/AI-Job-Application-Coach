from typing import Dict, Any
from langgraph.graph import StateGraph, END
import time

from app.graph.state import JobCoachState, INTENT_TYPES


class JobCoachWorkflow:
    """Main workflow orchestrator for the AI Job Application Coach."""
    
    def __init__(self):
        """Initialize the workflow with all agents and tools."""
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
            user_query = state.get("user_query", "").lower()
            intent = "unknown"
            confidence = 0.5
            
            # Simple intent classification (to be replaced with LLM)
            if any(word in user_query for word in ["resume", "cv", "review", "analyze"]):
                intent = "resume_analysis"
                confidence = 0.9
            elif any(word in user_query for word in ["interview", "practice", "questions", "mock"]):
                intent = "interview_practice"  
                confidence = 0.9
            elif any(word in user_query for word in ["job", "search", "find", "opportunities"]):
                intent = "job_search"
                confidence = 0.9
            elif any(word in user_query for word in ["advice", "help", "how", "tips", "guide"]):
                intent = "career_advice"
                confidence = 0.8
            elif any(word in user_query for word in ["application", "track", "status", "applied"]):
                intent = "application_tracking"
                confidence = 0.9
            
            # Update state
            agents_used = state.get("agents_used", [])
            agents_used.append("router")
            
            return {
                "intent": intent,
                "confidence": confidence,
                "agents_used": agents_used,
                "debug_info": {
                    "router_processing_time": time.time() - start_time,
                    "classification_method": "keyword_based"
                }
            }
            
        except Exception as e:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "error_message": f"Router agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["router"]
            }
    
    def _route_to_agent(self, state: JobCoachState) -> str:
        """Determine which agent to route to based on intent."""
        intent = state.get("intent", "unknown")
        
        if intent in ["resume_analysis", "resume_improvement"]:
            return "resume"
        elif intent in ["interview_practice", "interview_start", "interview_answer"]:
            return "interview"
        elif intent == "job_search":
            return "job_search"
        elif intent == "career_advice":
            return "knowledge"
        else:
            return "summary"  # Handle unknown intents in summary
    
    def _memory_load_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Load user profile and conversation history from memory."""
        start_time = time.time()
        
        try:
            # TODO: Implement actual memory loading from database
            # For now, return mock user profile
            
            user_id = state.get("user_id", 1)
            
            mock_profile = {
                "name": "Test User",
                "skills": ["Python", "FastAPI", "Machine Learning"],
                "experience_years": 5,
                "target_roles": ["Software Engineer", "ML Engineer"],
                "preferred_locations": ["Remote", "San Francisco"],
                "weak_areas": ["System design", "Salary negotiation"]
            }
            
            mock_history = [
                {
                    "message": "Previous resume review",
                    "agent_used": "resume", 
                    "created_at": "2024-02-05T10:00:00"
                }
            ]
            
            agents_used = state.get("agents_used", [])
            agents_used.append("memory_load")
            
            return {
                "user_profile": mock_profile,
                "conversation_history": mock_history,
                "agents_used": agents_used,
                "debug_info": {
                    "memory_load_time": time.time() - start_time
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Memory load error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["memory_load"]
            }
    
    def _resume_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Resume analysis and improvement agent."""
        start_time = time.time()
        
        try:
            # TODO: Implement actual resume analysis
            # For now, return mock analysis
            
            resume_text = state.get("resume_text", "")
            job_description = state.get("job_description", "")
            
            if not resume_text:
                return {
                    "error_message": "No resume text provided for analysis",
                    "agents_used": state.get("agents_used", []) + ["resume"]
                }
            
            mock_analysis = {
                "overall_score": 7.5,
                "strengths": [
                    "Clear work experience progression",
                    "Relevant technical skills listed",
                    "Good use of action verbs"
                ],
                "weaknesses": [
                    "Missing quantified achievements",
                    "Could improve professional summary",
                    "Some outdated technologies mentioned"
                ],
                "recommendations": [
                    "Add specific metrics to accomplishments (e.g., 'Improved performance by 25%')",
                    "Update skills section with current technologies",
                    "Tailor experience bullets to target role"
                ],
                "ats_compatibility": 8.0,
                "keyword_analysis": ["API development", "database optimization"] if job_description else None
            }
            
            agents_used = state.get("agents_used", [])
            agents_used.append("resume")
            
            return {
                "resume_analysis": mock_analysis,
                "agents_used": agents_used,
                "debug_info": {
                    "resume_processing_time": time.time() - start_time,
                    "resume_length": len(resume_text),
                    "job_description_provided": bool(job_description)
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Resume agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["resume"]
            }
    
    def _interview_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Interview practice and evaluation agent.""" 
        start_time = time.time()
        
        try:
            # TODO: Implement actual interview logic
            # For now, return mock interview data
            
            role = state.get("interview_role", "Software Engineer")
            level = state.get("interview_level", "mid")
            
            mock_questions = [
                {
                    "id": "q1",
                    "question": f"Tell me about your experience relevant to {role}.",
                    "type": "behavioral",
                    "difficulty": "easy",
                    "key_points": ["Relevant experience", "Role understanding", "Communication skills"]
                },
                {
                    "id": "q2", 
                    "question": "Describe a challenging project and how you overcame obstacles.",
                    "type": "behavioral",
                    "difficulty": "medium",
                    "key_points": ["Problem solving", "Persistence", "Technical skills", "Results"]
                }
            ]
            
            agents_used = state.get("agents_used", [])
            agents_used.append("interview")
            
            return {
                "interview_questions": mock_questions,
                "agents_used": agents_used,
                "debug_info": {
                    "interview_processing_time": time.time() - start_time,
                    "questions_generated": len(mock_questions)
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Interview agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["interview"]
            }
    
    def _job_search_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Job search and discovery agent."""
        start_time = time.time()
        
        try:
            # TODO: Implement actual job search
            # For now, return mock job results
            
            query = state.get("job_search_query", "Software Engineer")
            location = state.get("job_search_location", "Remote")
            
            mock_jobs = [
                {
                    "title": f"Senior {query}",
                    "company": "TechCorp",
                    "location": location,
                    "description": f"Seeking experienced {query} professional...",
                    "url": "https://example.com/job/1",
                    "salary_range": "$100K-$150K",
                    "remote_friendly": True,
                    "match_score": 0.85,
                    "source": "mock_api"
                },
                {
                    "title": f"Lead {query}",
                    "company": "InnovateInc", 
                    "location": "San Francisco, CA",
                    "description": f"Lead our {query} team...",
                    "url": "https://example.com/job/2",
                    "salary_range": "$120K-$180K",
                    "remote_friendly": False,
                    "match_score": 0.78,
                    "source": "mock_api"
                }
            ]
            
            agents_used = state.get("agents_used", [])
            agents_used.append("job_search")
            
            return {
                "job_results": mock_jobs,
                "agents_used": agents_used,
                "debug_info": {
                    "job_search_time": time.time() - start_time,
                    "results_found": len(mock_jobs),
                    "search_query": query,
                    "search_location": location
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Job search agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["job_search"]
            }
    
    def _knowledge_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Knowledge retrieval (RAG) agent for career advice."""
        start_time = time.time()
        
        try:
            # TODO: Implement actual RAG knowledge retrieval
            # For now, return mock career advice
            
            query = state.get("knowledge_query", state.get("user_query", ""))
            
            mock_answer = """
            Based on career best practices, here are key recommendations:
            
            1. **Build Your Network**: Attend industry events, join professional groups, and maintain active LinkedIn presence.
            
            2. **Continuous Learning**: Stay current with industry trends and invest in skill development.
            
            3. **Personal Branding**: Clearly articulate your unique value proposition and maintain consistent professional image.
            
            4. **Interview Preparation**: Practice common questions, prepare STAR method examples, and research company thoroughly.
            
            5. **Salary Negotiation**: Research market rates, document your achievements, and negotiate total compensation package.
            """
            
            mock_sources = [
                "Career Development Best Practices Guide",
                "Professional Networking Handbook", 
                "Interview Success Manual"
            ]
            
            agents_used = state.get("agents_used", [])
            agents_used.append("knowledge")
            
            return {
                "knowledge_answer": mock_answer,
                "knowledge_sources": mock_sources,
                "knowledge_context": "Career advice and best practices",
                "agents_used": agents_used,
                "debug_info": {
                    "knowledge_retrieval_time": time.time() - start_time,
                    "query_length": len(query),
                    "sources_found": len(mock_sources)
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Knowledge agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["knowledge"]
            }
    
    def _memory_save_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Save conversation and update user profile in memory."""
        start_time = time.time()
        
        try:
            # TODO: Implement actual memory persistence
            # For now, just log the save operation
            
            user_id = state.get("user_id", 1)
            session_id = state.get("session_id", "unknown")
            
            agents_used = state.get("agents_used", [])
            agents_used.append("memory_save")
            
            return {
                "agents_used": agents_used,
                "debug_info": {
                    "memory_save_time": time.time() - start_time,
                    "user_id": user_id,
                    "session_id": session_id,
                    "conversation_saved": True
                }
            }
            
        except Exception as e:
            return {
                "error_message": f"Memory save error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["memory_save"]
            }
    
    def _summary_agent(self, state: JobCoachState) -> Dict[str, Any]:
        """Synthesize final response from all agent outputs."""
        start_time = time.time()
        
        try:
            intent = state.get("intent", "unknown")
            error_message = state.get("error_message")
            
            # Handle error cases
            if error_message:
                response = f"I encountered an error while processing your request: {error_message}"
                return {
                    "response": response,
                    "session_complete": True,
                    "processing_time": time.time() - start_time,
                    "agents_used": state.get("agents_used", []) + ["summary"]
                }
            
            # Generate response based on intent and agent outputs
            response = ""
            
            if intent == "resume_analysis":
                analysis = state.get("resume_analysis", {})
                if analysis:
                    score = analysis.get("overall_score", 0)
                    strengths = analysis.get("strengths", [])
                    recommendations = analysis.get("recommendations", [])
                    
                    response = f"""
## Resume Analysis Results

**Overall Score: {score}/10**

### Strengths:
{chr(10).join('• ' + s for s in strengths[:3])}

### Key Recommendations:
{chr(10).join('• ' + r for r in recommendations[:3])}

Your resume shows good potential with some areas for improvement. Focus on quantifying your achievements and tailoring content to specific roles.
                    """.strip()
                else:
                    response = "I was unable to analyze your resume. Please ensure you've provided valid resume text."
            
            elif intent == "interview_practice":
                questions = state.get("interview_questions", [])
                if questions:
                    first_question = questions[0]
                    response = f"""
## Interview Practice Session Started

**Role**: {state.get('interview_role', 'Software Engineer')}
**Level**: {state.get('interview_level', 'mid')}

### First Question:
{first_question.get('question', 'Tell me about yourself.')}

**Type**: {first_question.get('type', 'behavioral')}
**Key Points to Cover**: {', '.join(first_question.get('key_points', []))}

Take your time to provide a thoughtful answer. I'll give you detailed feedback and follow up with additional questions.
                    """.strip()
                else:
                    response = "I'm ready to start your interview practice session. What role would you like to practice for?"
            
            elif intent == "job_search":
                jobs = state.get("job_results", [])
                if jobs:
                    job_list = ""
                    for i, job in enumerate(jobs[:3], 1):
                        job_list += f"""
{i}. **{job.get('title', 'Unknown Title')}** at {job.get('company', 'Unknown Company')}
   Location: {job.get('location', 'Unknown')}
   Salary: {job.get('salary_range', 'Not specified')}
   Match Score: {job.get('match_score', 0):.0%}
"""
                    
                    response = f"""
## Job Search Results

Found {len(jobs)} relevant opportunities:

{job_list.strip()}

These positions match your profile based on skills and experience. Would you like me to help you prepare application materials for any of these roles?
                    """.strip()
                else:
                    response = "I couldn't find any job opportunities matching your criteria. Try adjusting your search terms or location."
            
            elif intent == "career_advice":
                answer = state.get("knowledge_answer", "")
                sources = state.get("knowledge_sources", [])
                
                if answer:
                    response = f"""
## Career Advice

{answer}

---
*Sources: {', '.join(sources)}*
                    """.strip()
                else:
                    response = "I'd be happy to help with career advice. Could you be more specific about what you'd like guidance on?"
            
            else:
                response = "I understand you're looking for career assistance. I can help with resume reviews, interview practice, job searches, and career advice. What would you like to work on?"
            
            agents_used = state.get("agents_used", [])
            agents_used.append("summary")
            
            return {
                "response": response,
                "session_complete": True,
                "processing_time": time.time() - start_time,
                "agents_used": agents_used
            }
            
        except Exception as e:
            return {
                "response": f"I encountered an error while generating your response: {str(e)}",
                "session_complete": True,
                "error_message": f"Summary agent error: {str(e)}",
                "agents_used": state.get("agents_used", []) + ["summary"]
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
            "interview_questions": [],
            "interview_answers": [],
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