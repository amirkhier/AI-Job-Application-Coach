from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

class JobCoachState(TypedDict):
    """State schema for the AI Job Application Coach multi-agent system."""
    
    # Input and session management
    user_query: str
    user_id: int
    session_id: str
    
    # Router outputs
    intent: str
    confidence: float
    
    # Resume Agent data
    resume_text: Optional[str]
    job_description: Optional[str]
    resume_analysis: Optional[Dict[str, Any]]
    resume_suggestions: Optional[List[str]]
    
    # Interview Agent data
    interview_role: Optional[str]
    interview_level: Optional[str]
    interview_questions: List[Dict[str, Any]]
    interview_answers: List[Dict[str, Any]]
    interview_feedback: Optional[Dict[str, Any]]
    interview_session_id: Optional[str]
    
    # Job Search Agent data
    job_search_query: Optional[str]
    job_search_location: Optional[str]
    job_search_level: Optional[str]
    job_results: List[Dict[str, Any]]
    
    # Knowledge Agent (RAG) data
    knowledge_query: Optional[str]
    knowledge_context: Optional[str]
    knowledge_sources: List[str]
    knowledge_answer: Optional[str]
    
    # Memory Agent data
    user_profile: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    profile_updates: Optional[Dict[str, Any]]
    
    # Cross-agent communication
    agent_messages: List[Dict[str, Any]]
    shared_context: Optional[Dict[str, Any]]
    
    # Output and response
    response: str
    next_action: Optional[str]
    session_complete: bool
    
    # Metadata and tracking
    processing_time: float
    agents_used: List[str]
    error_message: Optional[str]
    debug_info: Optional[Dict[str, Any]]


class AgentMessage(TypedDict):
    """Schema for messages between agents."""
    from_agent: str
    to_agent: str
    message_type: str  # 'request', 'response', 'data', 'error'
    content: Any
    timestamp: float


class InterviewQuestion(TypedDict):
    """Schema for interview questions."""
    id: str
    question: str
    type: str  # 'behavioral', 'technical', 'situational'
    difficulty: str  # 'easy', 'medium', 'hard'
    key_points: List[str]
    expected_duration: Optional[int]  # in minutes


class InterviewAnswer(TypedDict):
    """Schema for interview answers."""
    question_id: str
    answer: str
    timestamp: float
    evaluation: Optional[Dict[str, Any]]


class JobListing(TypedDict):
    """Schema for job search results."""
    title: str
    company: str
    location: str
    description: str
    url: Optional[str]
    salary_range: Optional[str]
    remote_friendly: bool
    match_score: Optional[float]
    source: str


class ResumeAnalysis(TypedDict):
    """Schema for resume analysis results."""
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    ats_compatibility: float
    keyword_analysis: Optional[List[str]]
    sections_analysis: Optional[Dict[str, Dict[str, Any]]]


class UserProfile(TypedDict):
    """Schema for user profile data."""
    name: Optional[str]
    email: Optional[str]
    skills: List[str]
    experience_years: Optional[int]
    target_roles: List[str]
    preferred_locations: List[str]
    salary_range: Optional[Dict[str, int]]
    education: Optional[Dict[str, Any]]
    certifications: List[str]
    weak_areas: List[str]
    career_goals: Optional[str]


# Intent classification options
INTENT_TYPES = [
    "resume_analysis",
    "resume_improvement", 
    "interview_practice",
    "interview_start",
    "interview_answer",
    "job_search",
    "career_advice",
    "application_tracking",
    "general_question",
    "unknown"
]

# Agent names for routing
AGENT_NAMES = [
    "router",
    "resume",
    "interview", 
    "job_search",
    "knowledge",
    "memory",
    "summary"
]