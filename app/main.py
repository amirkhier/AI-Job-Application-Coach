from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import os
import time
import uuid
from datetime import datetime, date
from contextlib import asynccontextmanager

# Import our modules
from app.tools.database import init_db, close_db, get_db, DatabaseManager
from app.agents.resume import ResumeAgent
from app.agents.interview import InterviewAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.memory import MemoryAgent
from app.agents.job_search import JobSearchAgent

import json
import logging

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    database_connected: bool

class ResumeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Resume content in plain text")
    job_description: Optional[str] = Field(None, description="Target job description for tailored analysis")
    user_id: int = Field(1, description="User identifier")

class ATSCompatibility(BaseModel):
    score: float = Field(0.0, ge=0, le=10)
    issues: List[str] = []
    suggestions: List[str] = []

class KeywordAnalysis(BaseModel):
    present_keywords: List[str] = []
    missing_keywords: List[str] = []
    keyword_density_notes: str = ""

class SectionFeedback(BaseModel):
    contact_info: str = ""
    summary: str = ""
    experience: str = ""
    skills: str = ""
    education: str = ""

class ResumeResponse(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    ats_compatibility: ATSCompatibility
    keyword_analysis: KeywordAnalysis
    section_feedback: Optional[SectionFeedback] = None
    processing_time: float

class BulletImprovement(BaseModel):
    original: str
    improved: str
    reasoning: str

class ResumeImprovementResponse(BaseModel):
    improved_summary: str
    improved_bullets: List[BulletImprovement]
    additional_suggestions: List[str]
    priority_actions: List[str]
    processing_time: float

class InterviewStartRequest(BaseModel):
    role: str = Field(..., min_length=2, description="Target role for interview practice")
    level: str = Field("mid", pattern="^(junior|mid|senior|lead)$")
    question_count: int = Field(5, ge=1, le=10)
    user_id: int = Field(1)

class InterviewQuestion(BaseModel):
    id: str
    question: str
    type: str
    difficulty: str
    key_points: List[str]

class InterviewStartResponse(BaseModel):
    session_id: str
    role: str
    level: str
    first_question: InterviewQuestion
    total_questions: int

class InterviewAnswerRequest(BaseModel):
    session_id: str = Field(..., description="Interview session identifier")
    question_id: str = Field(..., description="Current question identifier") 
    answer: str = Field(..., min_length=10, description="User's answer to the question")

class InterviewFeedback(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    strength_areas: List[str]
    improvement_areas: List[str]
    specific_feedback: str
    suggested_improvement: str

class InterviewAnswerResponse(BaseModel):
    feedback: InterviewFeedback
    next_question: Optional[InterviewQuestion]
    session_complete: bool
    session_summary: Optional[Dict[str, Any]]

class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Career-related question")
    user_id: int = Field(1)

class KnowledgeQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    relevance_score: float
    related_topics: List[str]

class JobSearchRequest(BaseModel):
    query: str = Field(..., description="Job search keywords")
    location: str = Field(..., description="Preferred job location")
    experience_level: str = Field("mid", pattern="^(junior|mid|senior|lead)$")
    remote_ok: bool = Field(True)
    count: int = Field(5, ge=1, le=10, description="Number of results")
    user_id: int = Field(1)

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: Optional[str] = None
    salary_range: Optional[str] = None
    remote_friendly: bool = False
    match_score: Optional[float] = None
    experience_level: Optional[str] = None
    key_skills: List[str] = []

class NearbyCompany(BaseModel):
    name: str
    type: str
    distance_m: int
    address: str = ""

class LocationInfo(BaseModel):
    city: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    display_name: str = ""
    country: str = ""
    found: bool = False

class JobSearchResponse(BaseModel):
    jobs: List[JobListing]
    total_found: int
    search_query: str
    location: str
    location_info: Optional[LocationInfo] = None
    nearby_companies: List[NearbyCompany] = []
    processing_time: float = 0.0

class ApplicationCreateRequest(BaseModel):
    company_name: str
    position_title: str
    job_url: Optional[str]
    application_date: Optional[date]
    notes: Optional[str]
    user_id: int = Field(1)

class ApplicationResponse(BaseModel):
    id: int
    company_name: str
    position_title: str
    job_url: Optional[str]
    status: str
    application_date: date
    follow_up_date: Optional[date]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

class ApplicationUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, pattern="^(applied|interviewing|offer|rejected|withdrawn)$")
    notes: Optional[str]
    follow_up_date: Optional[date]

class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime]

class UserContextResponse(BaseModel):
    user_profile: Dict[str, Any]
    recent_conversations: List[Dict[str, Any]]
    context_summary: Dict[str, Any]
    history_count: int

class ConversationAnalysisResponse(BaseModel):
    insights: List[str]
    patterns: List[str]
    recommendations: List[str]
    conversation_count: int
    agent_usage: Dict[str, int]

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting AI Job Application Coach...")
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down AI Job Application Coach...")
    try:
        close_db()
        print("Database connection closed")
    except Exception as e:
        print(f"Warning: Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="AI Job Application Coach",
    description="Multi-agent career coaching system powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database
def get_database():
    """Dependency to get database instance."""
    return get_db()

# Initialise agents
resume_agent = ResumeAgent()
interview_agent = InterviewAgent()
knowledge_agent = KnowledgeAgent()
memory_agent = MemoryAgent()
job_search_agent = JobSearchAgent()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(db: DatabaseManager = Depends(get_database)):
    """Health check endpoint to verify service status."""
    database_connected = False
    try:
        db.ensure_connection()
        # Test database with simple query
        result = db.execute_query("SELECT 1 as test")
        database_connected = result is not None
    except Exception as e:
        print(f"Database health check failed: {e}")
    
    return HealthResponse(
        status="healthy",
        service="AI Job Application Coach",
        timestamp=datetime.now().isoformat(),
        database_connected=database_connected
    )

# Resume analysis endpoints
@app.post("/resume", response_model=ResumeResponse)
async def analyze_resume(request: ResumeRequest, db: DatabaseManager = Depends(get_database)):
    """Analyse a resume and return LLM-powered structured feedback."""
    try:
        # Run real Resume Agent analysis
        analysis = resume_agent.analyze_resume(
            resume_text=request.resume_text,
            job_description=request.job_description or "",
        )

        # Persist conversation with intelligent analysis (non-fatal)
        session_id = str(uuid.uuid4())
        try:
            analysis_summary = f"Resume analysis completed. Overall score: {analysis.get('overall_score')}/10"
            memory_agent.save_conversation_with_analysis(
                user_id=request.user_id,
                session_id=session_id,
                user_message=f"Analyze my resume ({len(request.resume_text)} chars)" + 
                           (f" for {request.job_description[:100]}..." if request.job_description else ""),
                agent_response=analysis_summary,
                agent_type="resume",
                intent="resume_analysis"
            )
            
            # Also update profile with any insights
            memory_agent.update_profile_from_conversation(
                user_id=request.user_id,
                user_message=f"Resume analysis request targeting: {request.job_description[:200] if request.job_description else 'general positions'}",
                agent_response=analysis_summary
            )
        except Exception as db_err:
            logger.warning("Memory agent save failed (non-fatal): %s", db_err)

        # Map agent dict → Pydantic response models
        ats_raw = analysis.get("ats_compatibility", {})
        kw_raw = analysis.get("keyword_analysis", {})
        sf_raw = analysis.get("section_feedback", {})

        return ResumeResponse(
            overall_score=analysis.get("overall_score", 0.0),
            strengths=analysis.get("strengths", []),
            weaknesses=analysis.get("weaknesses", []),
            recommendations=analysis.get("recommendations", []),
            ats_compatibility=ATSCompatibility(
                score=ats_raw.get("score", 0.0) if isinstance(ats_raw, dict) else 0.0,
                issues=ats_raw.get("issues", []) if isinstance(ats_raw, dict) else [],
                suggestions=ats_raw.get("suggestions", []) if isinstance(ats_raw, dict) else [],
            ),
            keyword_analysis=KeywordAnalysis(
                present_keywords=kw_raw.get("present_keywords", []) if isinstance(kw_raw, dict) else [],
                missing_keywords=kw_raw.get("missing_keywords", []) if isinstance(kw_raw, dict) else [],
                keyword_density_notes=kw_raw.get("keyword_density_notes", "") if isinstance(kw_raw, dict) else "",
            ),
            section_feedback=SectionFeedback(**sf_raw) if isinstance(sf_raw, dict) and sf_raw else None,
            processing_time=analysis.get("processing_time", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resume analysis endpoint failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {str(e)}")

@app.post("/resume/audit", response_model=AsyncTaskResponse) 
async def request_detailed_resume_audit(request: ResumeRequest, background_tasks: BackgroundTasks):
    """Request detailed resume audit as background task."""
    task_id = str(uuid.uuid4())
    
    # TODO: Implement Celery task for detailed audit
    # For now, return mock task response
    
    return AsyncTaskResponse(
        task_id=task_id,
        status="queued",
        message="Resume audit task queued for processing",
        estimated_completion=datetime.now()
    )

@app.post("/resume/improve", response_model=ResumeImprovementResponse)
async def improve_resume(request: ResumeRequest, db: DatabaseManager = Depends(get_database)):
    """Generate concrete improvement suggestions with rewritten examples."""
    try:
        improvements = resume_agent.suggest_improvements(
            resume_text=request.resume_text,
            job_description=request.job_description or "",
        )

        # Persist conversation (non-fatal)
        session_id = str(uuid.uuid4())
        try:
            db.save_conversation(
                user_id=request.user_id,
                session_id=session_id,
                message=f"Resume improvement request: {len(request.resume_text)} characters",
                intent="resume_improvement",
                agent_used="resume",
                metadata={"job_description_provided": bool(request.job_description)},
            )
        except Exception as db_err:
            logger.warning("Failed to save conversation: %s", db_err)

        return ResumeImprovementResponse(
            improved_summary=improvements.get("improved_summary", ""),
            improved_bullets=[
                BulletImprovement(**b) for b in improvements.get("improved_bullets", [])
            ],
            additional_suggestions=improvements.get("additional_suggestions", []),
            priority_actions=improvements.get("priority_actions", []),
            processing_time=improvements.get("processing_time", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Resume improvement endpoint failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Resume improvement failed: {str(e)}")

# Interview practice endpoints
@app.post("/interview/start", response_model=InterviewStartResponse)
async def start_interview_session(request: InterviewStartRequest, db: DatabaseManager = Depends(get_database)):
    """Start a new mock-interview session with LLM-generated questions."""
    try:
        session_id = str(uuid.uuid4())

        # Create session row in database
        try:
            db.create_interview_session(
                user_id=request.user_id,
                session_id=session_id,
                role=request.role,
                level=request.level,
            )
        except Exception as db_err:
            logger.warning("Failed to create interview session in DB: %s", db_err)

        # Generate all questions up-front via the Interview Agent
        questions = interview_agent.generate_questions(
            role=request.role,
            level=request.level,
            count=request.question_count,
        )

        # Persist questions in the session row
        try:
            db.update_interview_session(session_id=session_id, questions=questions)
        except Exception as db_err:
            logger.warning("Failed to persist questions to DB: %s", db_err)

        first_q = questions[0]
        first_question = InterviewQuestion(
            id=first_q["id"],
            question=first_q["question"],
            type=first_q.get("type", "behavioral"),
            difficulty=first_q.get("difficulty", "easy"),
            key_points=first_q.get("key_points", []),
        )

        return InterviewStartResponse(
            session_id=session_id,
            role=request.role,
            level=request.level,
            first_question=first_question,
            total_questions=len(questions),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start interview session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start interview session: {str(e)}")

@app.post("/interview/answer", response_model=InterviewAnswerResponse)
async def submit_interview_answer(request: InterviewAnswerRequest, db: DatabaseManager = Depends(get_database)):
    """Submit an answer, receive LLM-powered evaluation, and advance the session."""
    try:
        # ---- Load session ------------------------------------------------
        session = db.get_interview_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")

        questions: List[Dict] = session.get("questions", [])
        answers: List[Dict] = session.get("answers", [])
        role = session.get("role", "Software Engineer")
        level = session.get("level", "mid")

        # Find the question being answered
        current_q = next((q for q in questions if q["id"] == request.question_id), None)
        if not current_q:
            raise HTTPException(status_code=404, detail=f"Question {request.question_id} not found in session")

        # ---- Evaluate answer via Interview Agent -------------------------
        evaluation = interview_agent.evaluate_answer(
            question=current_q,
            answer=request.answer,
            role=role,
            level=level,
        )

        feedback = InterviewFeedback(
            overall_score=evaluation.get("overall_score", 5.0),
            strength_areas=evaluation.get("strength_areas", []),
            improvement_areas=evaluation.get("improvement_areas", []),
            specific_feedback=evaluation.get("specific_feedback", ""),
            suggested_improvement=evaluation.get("suggested_improvement", ""),
        )

        # ---- Record answer -----------------------------------------------
        answers.append({
            "question_id": request.question_id,
            "answer": request.answer,
            "evaluation": evaluation,
        })

        # ---- Determine next question or end session ----------------------
        answered_ids = {a["question_id"] for a in answers}
        remaining = [q for q in questions if q["id"] not in answered_ids]

        session_complete = len(remaining) == 0
        next_question = None
        session_summary = None

        if not session_complete:
            nq = remaining[0]
            next_question = InterviewQuestion(
                id=nq["id"],
                question=nq["question"],
                type=nq.get("type", "behavioral"),
                difficulty=nq.get("difficulty", "medium"),
                key_points=nq.get("key_points", []),
            )
        else:
            # Generate session summary via Interview Agent
            try:
                session_summary = interview_agent.generate_session_summary(
                    questions=questions,
                    answers_with_feedback=answers,
                    role=role,
                    level=level,
                )
            except Exception as sum_err:
                logger.warning("Session summary generation failed: %s", sum_err)
                session_summary = {"error": str(sum_err)}

        # ---- Persist to DB (non-fatal) -----------------------------------
        try:
            scores = [
                a["evaluation"]["overall_score"]
                for a in answers
                if a.get("evaluation") and a["evaluation"].get("overall_score")
            ]
            avg_score = round(sum(scores) / len(scores), 2) if scores else None

            db.update_interview_session(
                session_id=request.session_id,
                answers=answers,
                feedback=session_summary if session_complete else None,
                score=avg_score,
                completed=session_complete,
            )
        except Exception as db_err:
            logger.warning("Failed to persist interview answer: %s", db_err)

        return InterviewAnswerResponse(
            feedback=feedback,
            next_question=next_question,
            session_complete=session_complete,
            session_summary=session_summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process interview answer: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")


@app.get("/interview/questions/{job_title}")
async def get_interview_questions(
    job_title: str,
    level: str = "mid",
    count: int = 5,
):
    """Generate interview questions for a role without starting a full session."""
    try:
        questions = interview_agent.generate_questions(
            role=job_title,
            level=level,
            count=min(count, 10),
        )
        return {"role": job_title, "level": level, "questions": questions}
    except Exception as e:
        logger.error("Question generation endpoint failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


# Knowledge query endpoint
@app.post("/ask", response_model=KnowledgeQueryResponse)
async def ask_career_question(request: KnowledgeQueryRequest, db: DatabaseManager = Depends(get_database)):
    """Ask career-related questions and get RAG-powered advice from the knowledge base."""
    try:
        # Run RAG-powered Knowledge Agent
        result = knowledge_agent.answer_question(request.query)

        # Save conversation with intelligent analysis (non-fatal)
        session_id = str(uuid.uuid4())
        try:
            memory_agent.save_conversation_with_analysis(
                user_id=request.user_id,
                session_id=session_id,
                user_message=request.query,
                agent_response=result["answer"],
                agent_type="knowledge",
                intent="career_advice"
            )
            
            # Update profile with insights from Q&A
            memory_agent.update_profile_from_conversation(
                user_id=request.user_id,
                user_message=request.query,
                agent_response=result["answer"]
            )
        except Exception as db_err:
            logger.warning("Memory agent save failed (non-fatal): %s", db_err)

        return KnowledgeQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            relevance_score=result["relevance_score"],
            related_topics=result["related_topics"],
        )

    except Exception as e:
        logger.error("Knowledge query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Knowledge query failed: {str(e)}")

# Job search endpoints  
@app.post("/jobs/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest, db: DatabaseManager = Depends(get_database)):
    """Search for job opportunities using geolocation + LLM-powered results."""
    try:
        # Run the Job Search Agent
        result = job_search_agent.search_jobs(
            query=request.query,
            location=request.location,
            experience_level=request.experience_level,
            remote_ok=request.remote_ok,
            count=request.count,
        )

        # Map raw dicts → Pydantic models
        job_listings = [
            JobListing(
                title=j.get("title", "Unknown"),
                company=j.get("company", "Unknown"),
                location=j.get("location", ""),
                description=j.get("description", ""),
                url=j.get("url"),
                salary_range=j.get("salary_range"),
                remote_friendly=j.get("remote_friendly", False),
                match_score=j.get("match_score"),
                experience_level=j.get("experience_level"),
                key_skills=j.get("key_skills", []),
            )
            for j in result.get("jobs", [])
        ]

        # Map location info
        loc_raw = result.get("location_info", {})
        location_info = LocationInfo(
            city=loc_raw.get("city", request.location),
            lat=loc_raw.get("lat"),
            lon=loc_raw.get("lon"),
            display_name=loc_raw.get("display_name", ""),
            country=loc_raw.get("country", ""),
            found=loc_raw.get("found", False),
        )

        # Map nearby companies
        nearby = [
            NearbyCompany(
                name=c.get("name", "Unknown"),
                type=c.get("type", "office"),
                distance_m=c.get("distance_m", 0),
                address=c.get("address", ""),
            )
            for c in result.get("nearby_companies", [])
        ]

        # Save via Memory Agent (non-fatal)
        session_id = str(uuid.uuid4())
        try:
            memory_agent.save_conversation_with_analysis(
                user_id=request.user_id,
                session_id=session_id,
                user_message=f"Job search: {request.query} in {request.location} ({request.experience_level})",
                agent_response=f"Found {len(job_listings)} positions",
                agent_type="job_search",
                intent="job_search"
            )
        except Exception as db_err:
            logger.warning("Memory agent save failed (non-fatal): %s", db_err)

        return JobSearchResponse(
            jobs=job_listings,
            total_found=len(job_listings),
            search_query=request.query,
            location=request.location,
            location_info=location_info,
            nearby_companies=nearby,
            processing_time=result.get("processing_time", 0.0),
        )

    except Exception as e:
        logger.error("Job search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@app.get("/jobs/location/{city}")
async def get_city_info(city: str):
    """Geocode a city and find nearby tech companies/offices."""
    try:
        location_info = job_search_agent.get_city_center(city)
        nearby: List[Dict[str, Any]] = []

        if location_info.get("lat") and location_info.get("lon"):
            nearby = job_search_agent.find_nearby_offices(
                lat=location_info["lat"],
                lon=location_info["lon"],
                radius=5000,
            )

        return {
            "location": location_info,
            "nearby_companies": nearby[:10],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Location lookup failed: {str(e)}")


@app.post("/jobs/match")
async def match_jobs_to_profile(
    request: JobSearchRequest,
    db: DatabaseManager = Depends(get_database),
):
    """Search jobs and score them against the user's profile."""
    try:
        # Load user profile for matching
        user_context = memory_agent.load_user_context(
            user_id=request.user_id,
            interaction_type="job_search",
        )
        user_profile = user_context.get("profile", {})

        result = job_search_agent.search_jobs_with_matching(
            query=request.query,
            location=request.location,
            experience_level=request.experience_level,
            remote_ok=request.remote_ok,
            user_profile=user_profile if user_profile else None,
            count=request.count,
        )

        return result

    except Exception as e:
        logger.error("Job matching failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(e)}")

# Application tracking endpoints
@app.post("/applications", response_model=ApplicationResponse)
async def create_application(request: ApplicationCreateRequest, db: DatabaseManager = Depends(get_database)):
    """Create a new job application record."""
    try:
        application_date = request.application_date or datetime.now().date()
        
        application_id = db.create_application(
            user_id=request.user_id,
            company_name=request.company_name,
            position_title=request.position_title,
            job_url=request.job_url,
            application_date=application_date,
            notes=request.notes
        )
        
        if not application_id:
            raise HTTPException(status_code=500, detail="Failed to create application")
        
        # Return mock response (in real implementation, fetch from DB)
        return ApplicationResponse(
            id=application_id,
            company_name=request.company_name,
            position_title=request.position_title,
            job_url=request.job_url,
            status="applied",
            application_date=application_date,
            follow_up_date=None,
            notes=request.notes,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create application: {str(e)}")

@app.get("/applications", response_model=List[ApplicationResponse])
async def get_applications(user_id: int = 1, status: Optional[str] = None, db: DatabaseManager = Depends(get_database)):
    """Get user's job applications, optionally filtered by status."""
    try:
        applications = db.get_applications(user_id=user_id, status=status)
        
        # Convert to response model (handle potential None dates)
        response_applications = []
        for app in applications:
            response_applications.append(ApplicationResponse(
                id=app['id'],
                company_name=app['company_name'],
                position_title=app['position_title'], 
                job_url=app['job_url'],
                status=app['status'],
                application_date=app['application_date'],
                follow_up_date=app['follow_up_date'],
                notes=app['notes'],
                created_at=app['created_at'],
                updated_at=app['updated_at']
            ))
        
        return response_applications
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve applications: {str(e)}")

@app.put("/applications/{application_id}", response_model=ApplicationResponse)
async def update_application(application_id: int, request: ApplicationUpdateRequest, db: DatabaseManager = Depends(get_database)):
    """Update job application status and details."""
    try:
        # TODO: Implement application update
        # For now, return mock response
        
        success = db.update_application_status(
            application_id=application_id,
            status=request.status or "applied",
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Return mock updated application
        return ApplicationResponse(
            id=application_id,
            company_name="Example Company",
            position_title="Software Engineer",
            job_url=None,
            status=request.status or "applied",
            application_date=date.today(),
            follow_up_date=request.follow_up_date,
            notes=request.notes,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update application: {str(e)}")

# Async task result endpoint
@app.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """Get result of asynchronous task."""
    # TODO: Implement Celery task result retrieval
    # For now, return mock result
    return {
        "task_id": task_id,
        "status": "completed",
        "result": {
            "message": "Task completed successfully",
            "data": {}
        }
    }

# User profile endpoints
@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int, db: DatabaseManager = Depends(get_database)):
    """Get user profile data."""
    try:
        user = db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": user["id"],
            "email": user["email"],
            "profile_data": user.get("profile_data", {}),
            "preferences": user.get("preferences", {}),
            "created_at": user["created_at"],
            "updated_at": user["updated_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

# Memory Agent endpoints
@app.get("/user/{user_id}/context", response_model=UserContextResponse)
async def get_user_context(
    user_id: int, 
    interaction_type: str = "general",
    history_limit: int = 5,
    db: DatabaseManager = Depends(get_database)
):
    """Get user context and conversation history for other agents."""
    try:
        context = memory_agent.load_user_context(
            user_id=user_id,
            interaction_type=interaction_type,
            history_limit=history_limit
        )
        
        return UserContextResponse(
            user_profile=context["profile"],
            recent_conversations=context["recent_conversations"],
            context_summary=context["context_summary"],
            history_count=context["history_count"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load user context: {str(e)}")

@app.get("/user/{user_id}/insights", response_model=ConversationAnalysisResponse)  
async def get_conversation_insights(
    user_id: int,
    session_id: Optional[str] = None,
    days_back: int = 7,
    db: DatabaseManager = Depends(get_database)
):
    """Get analytical insights from user's conversation patterns."""
    try:
        insights = memory_agent.get_conversation_insights(
            user_id=user_id,
            session_id=session_id,
            days_back=days_back
        )
        
        return ConversationAnalysisResponse(
            insights=insights["insights"],
            patterns=insights["patterns"], 
            recommendations=insights["recommendations"],
            conversation_count=insights["conversation_count"],
            agent_usage=insights.get("agent_usage", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation insights: {str(e)}")

@app.post("/user/{user_id}/profile/update")
async def update_profile_from_conversation(
    user_id: int,
    user_message: str,
    agent_response: str,
    db: DatabaseManager = Depends(get_database)
):
    """Update user profile based on conversation insights."""
    try:
        result = memory_agent.update_profile_from_conversation(
            user_id=user_id,
            user_message=user_message,
            agent_response=agent_response
        )
        
        return {
            "updated": result["updated"],
            "new_insights": result["new_insights"],
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port, reload=True)