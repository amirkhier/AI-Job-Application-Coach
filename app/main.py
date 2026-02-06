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

class ResumeResponse(BaseModel):
    overall_score: float
    strengths: List[str]
    weaknesses: List[str] 
    recommendations: List[str]
    ats_compatibility: float
    keyword_analysis: Optional[List[str]]
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
    experience_level: str = Field("mid", pattern="^(entry|mid|senior)$")
    remote_ok: bool = Field(True)
    user_id: int = Field(1)

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: Optional[str]
    salary_range: Optional[str]
    remote_friendly: bool

class JobSearchResponse(BaseModel):
    jobs: List[JobListing]
    total_found: int
    search_query: str
    location: str

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
    """Analyze resume and provide improvement feedback."""
    start_time = time.time()
    
    try:
        # TODO: Implement resume analysis using LangGraph workflow
        # For now, return a mock response
        
        # Save conversation to database
        session_id = str(uuid.uuid4())
        db.save_conversation(
            user_id=request.user_id,
            session_id=session_id,
            message=f"Resume analysis request: {len(request.resume_text)} characters",
            intent="resume_analysis",
            agent_used="resume",
            metadata={"job_description_provided": bool(request.job_description)}
        )
        
        processing_time = time.time() - start_time
        
        return ResumeResponse(
            overall_score=7.5,
            strengths=["Clear work experience section", "Relevant technical skills"],
            weaknesses=["Missing quantified achievements", "Could improve summary section"],
            recommendations=[
                "Add specific metrics to accomplishments",
                "Include relevant keywords from job description",
                "Improve professional summary with value proposition"
            ],
            ats_compatibility=8.0,
            keyword_analysis=["Python", "APIs", "databases"] if request.job_description else None,
            processing_time=processing_time
        )
        
    except Exception as e:
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

# Interview practice endpoints
@app.post("/interview/start", response_model=InterviewStartResponse)
async def start_interview_session(request: InterviewStartRequest, db: DatabaseManager = Depends(get_database)):
    """Start a new interview practice session."""
    try:
        session_id = str(uuid.uuid4())
        
        # Create interview session in database
        db.create_interview_session(
            user_id=request.user_id,
            session_id=session_id,
            role=request.role,
            level=request.level
        )
        
        # TODO: Implement interview question generation using LangGraph
        # For now, return a mock response
        
        first_question = InterviewQuestion(
            id="q1",
            question=f"Tell me about your experience in {request.role} and what interests you about this position.",
            type="behavioral", 
            difficulty="easy",
            key_points=["Relevant experience", "Motivation", "Role understanding"]
        )
        
        return InterviewStartResponse(
            session_id=session_id,
            role=request.role,
            level=request.level,
            first_question=first_question,
            total_questions=request.question_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start interview session: {str(e)}")

@app.post("/interview/answer", response_model=InterviewAnswerResponse)
async def submit_interview_answer(request: InterviewAnswerRequest, db: DatabaseManager = Depends(get_database)):
    """Submit answer to interview question and get feedback."""
    try:
        # Get interview session
        session = db.get_interview_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        # TODO: Implement answer evaluation using LangGraph
        # For now, return mock feedback
        
        feedback = InterviewFeedback(
            overall_score=7.0,
            strength_areas=["Clear communication", "Relevant examples"],
            improvement_areas=["Could provide more specific metrics", "Structure using STAR method"],
            specific_feedback="Your answer shows good understanding but could benefit from more concrete examples.",
            suggested_improvement="Try structuring your response using the STAR method: Situation, Task, Action, Result."
        )
        
        # Mock next question (in real implementation, check if session complete)
        next_question = InterviewQuestion(
            id="q2", 
            question="Describe a challenging project you worked on and how you overcame obstacles.",
            type="behavioral",
            difficulty="medium",
            key_points=["Problem-solving", "Persistence", "Technical skills", "Results"]
        )
        
        return InterviewAnswerResponse(
            feedback=feedback,
            next_question=next_question,
            session_complete=False,
            session_summary=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")

# Knowledge query endpoint
@app.post("/ask", response_model=KnowledgeQueryResponse)
async def ask_career_question(request: KnowledgeQueryRequest, db: DatabaseManager = Depends(get_database)):
    """Ask career-related questions and get advice from knowledge base."""
    try:
        # Save conversation
        session_id = str(uuid.uuid4())
        db.save_conversation(
            user_id=request.user_id,
            session_id=session_id,
            message=request.query,
            intent="career_advice",
            agent_used="knowledge"
        )
        
        # TODO: Implement RAG-based knowledge retrieval using LangGraph
        # For now, return mock response
        
        return KnowledgeQueryResponse(
            answer="Based on industry best practices, I recommend focusing on building your professional network, updating your skills regularly, and clearly articulating your value proposition to potential employers.",
            sources=["Career Best Practices Guide", "Interview Tips Manual"],
            relevance_score=0.85,
            related_topics=["networking", "skill development", "personal branding"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge query failed: {str(e)}")

# Job search endpoints  
@app.post("/jobs/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest, db: DatabaseManager = Depends(get_database)):
    """Search for job opportunities based on criteria."""
    try:
        # TODO: Implement job search using LangGraph workflow
        # For now, return mock results
        
        mock_jobs = [
            JobListing(
                title=f"Senior {request.query}",
                company="TechCorp Inc.",
                location=request.location,
                description=f"Seeking experienced {request.query} professional...",
                url="https://example.com/job/1",
                salary_range="$80K-$120K",
                remote_friendly=request.remote_ok
            ),
            JobListing(
                title=f"{request.experience_level.title()} {request.query}",
                company="StartupXYZ",
                location="Remote",
                description=f"Join our growing team as a {request.query}...",
                url="https://example.com/job/2", 
                salary_range="$70K-$100K",
                remote_friendly=True
            )
        ]
        
        return JobSearchResponse(
            jobs=mock_jobs,
            total_found=len(mock_jobs),
            search_query=request.query,
            location=request.location
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")

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

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port, reload=True)