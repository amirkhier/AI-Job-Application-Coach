# API Reference

Base URL: `http://localhost:8000`

> Interactive Swagger UI: `http://localhost:8000/docs`  
> OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Authentication

When `API_KEY` is configured, all endpoints (except `/health`) require:

```
X-API-Key: <your-api-key>
```

Rate limit: **60 requests/minute** per IP (configurable via `RATE_LIMIT_PER_MINUTE`).

---

## Health

### `GET /health`

Returns service health status and dependency checks.

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "service": "AI Job Application Coach",
  "timestamp": "2026-03-02T20:00:00.000Z",
  "database_connected": true,
  "checks": {
    "database": true,
    "redis": false,
    "chromadb": true
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` or `degraded` |
| `database_connected` | boolean | MySQL connectivity |
| `checks` | object | Per-dependency status |

---

## Resume

### `POST /resume`

Analyse a resume and return structured scoring, ATS compatibility, keyword analysis, and section feedback.

**Request Body** (`ResumeRequest`):
```json
{
  "resume_text": "Jane Smith\nSenior Backend Engineer...",
  "job_description": "Senior Backend Engineer at FinTech...",
  "user_id": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resume_text` | string | Yes (min 50 chars) | Plain-text resume |
| `job_description` | string | No | Target job posting |
| `user_id` | integer | No (default 1) | User identifier |

**Response** `200 OK` (`ResumeResponse`):
```json
{
  "overall_score": 7.5,
  "strengths": ["Quantified achievements", "Strong technical skills"],
  "weaknesses": ["Missing professional summary"],
  "recommendations": ["Add a concise summary paragraph"],
  "ats_compatibility": {
    "score": 8.0,
    "issues": [],
    "suggestions": ["Add more role-specific keywords"]
  },
  "keyword_analysis": {
    "present_keywords": ["Python", "AWS", "Docker"],
    "missing_keywords": ["CI/CD", "Terraform"],
    "keyword_density_notes": "Good keyword coverage"
  },
  "section_feedback": {
    "contact_info": "Complete",
    "summary": "Missing — should add",
    "experience": "Strong with metrics",
    "skills": "Well-organized",
    "education": "Adequate"
  },
  "processing_time": 3.42
}
```

### `POST /resume/improve`

Generate specific rewrite suggestions based on a prior analysis.

**Request Body** (`ResumeRequest`):
```json
{
  "resume_text": "...",
  "job_description": "...",
  "user_id": 1
}
```

**Response** `200 OK` (`ResumeImprovementResponse`):
```json
{
  "improved_summary": "Results-driven senior backend engineer with 7+ years...",
  "improved_bullets": [
    {
      "original": "Led migration of monolithic app",
      "improved": "Led cloud migration of monolith to 12 microservices, reducing deploy time by 60% and improving uptime to 99.95%",
      "reasoning": "Added specific metrics and scope"
    }
  ],
  "additional_suggestions": ["Add certifications section"],
  "priority_actions": ["Rewrite summary", "Quantify all bullet points"],
  "processing_time": 4.1
}
```

### `POST /resume/audit`

Submit a resume for asynchronous full audit (Celery task).

**Response** `202 Accepted` (`AsyncTaskResponse`):
```json
{
  "task_id": "abc123-...",
  "status": "queued",
  "message": "Resume audit queued",
  "estimated_completion": "2026-03-02T20:05:00Z"
}
```

---

## Interview

### `POST /interview/start`

Start a new mock interview session.

**Request Body** (`InterviewStartRequest`):
```json
{
  "role": "Backend Engineer",
  "level": "senior",
  "question_count": 5,
  "user_id": 1
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `role` | string | Yes (min 2 chars) | Target job role |
| `level` | string | No (default `mid`) | `junior` / `mid` / `senior` / `lead` |
| `question_count` | integer | No (default 5) | 1–10 |
| `user_id` | integer | No (default 1) | User identifier |

**Response** `200 OK` (`InterviewStartResponse`):
```json
{
  "session_id": "uuid-...",
  "role": "Backend Engineer",
  "level": "senior",
  "first_question": {
    "id": "q1",
    "question": "Tell me about a time you led a complex migration project.",
    "type": "behavioral",
    "difficulty": "hard",
    "key_points": ["Leadership", "Technical complexity", "Outcome"]
  },
  "total_questions": 5
}
```

### `POST /interview/answer`

Submit an answer to an interview question and receive evaluation.

**Request Body** (`InterviewAnswerRequest`):
```json
{
  "session_id": "uuid-...",
  "question_id": "q1",
  "answer": "At my previous role, I was tasked with migrating..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | Yes | Session from `/interview/start` |
| `question_id` | string | Yes | Question ID being answered |
| `answer` | string | Yes (min 10 chars) | Candidate's answer |

**Response** `200 OK` (`InterviewAnswerResponse`):
```json
{
  "feedback": {
    "overall_score": 7.5,
    "strength_areas": ["Clear STAR structure", "Specific metrics"],
    "improvement_areas": ["Could elaborate on team dynamics"],
    "specific_feedback": "Good use of quantified results...",
    "suggested_improvement": "Consider adding more context about stakeholder management"
  },
  "next_question": { "id": "q2", "question": "...", "type": "technical", "difficulty": "medium", "key_points": [] },
  "session_complete": false,
  "session_summary": null
}
```

### `GET /interview/questions/{job_title}`

Quick question generation without session management.

**Path Parameters**: `job_title` (string)  
**Query Parameters**: `level` (default `mid`), `count` (default 5)

**Response** `200 OK`: Array of `InterviewQuestion` objects.

### `POST /interview/report`

Asynchronous interview performance report generation.

**Response** `202 Accepted`: `AsyncTaskResponse`

---

## Knowledge

### `POST /ask`

Ask a career-related question. Uses RAG to ground answers in the built-in career guide corpus.

**Request Body** (`KnowledgeQueryRequest`):
```json
{
  "query": "How should I negotiate my salary after receiving an offer?",
  "user_id": 1
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes (min 5 chars) | Career question |
| `user_id` | integer | No (default 1) | User identifier |

**Response** `200 OK` (`KnowledgeQueryResponse`):
```json
{
  "answer": "When negotiating salary, start by researching market rates...",
  "sources": ["salary_negotiation.md", "industry_insights.md"],
  "relevance_score": 0.87,
  "related_topics": ["benefits negotiation", "equity packages", "counter-offers"]
}
```

---

## Job Search

### `POST /jobs/search`

Search for job listings (LLM-generated based on query context).

**Request Body** (`JobSearchRequest`):
```json
{
  "query": "Python backend developer",
  "location": "Tel Aviv",
  "experience_level": "senior",
  "remote_ok": true,
  "count": 5,
  "user_id": 1
}
```

**Response** `200 OK` (`JobSearchResponse`):
```json
{
  "jobs": [
    {
      "title": "Senior Python Developer",
      "company": "TechCorp",
      "location": "Tel Aviv, Israel",
      "description": "Build scalable backend services...",
      "url": null,
      "salary_range": "$120k-$160k",
      "remote_friendly": true,
      "match_score": null,
      "experience_level": "senior",
      "key_skills": ["Python", "FastAPI", "AWS"]
    }
  ],
  "total_found": 5,
  "search_query": "Python backend developer",
  "location": "Tel Aviv",
  "location_info": {
    "city": "Tel Aviv",
    "lat": 32.0853,
    "lon": 34.7818,
    "display_name": "Tel Aviv-Yafo, Israel",
    "country": "Israel",
    "found": true
  },
  "nearby_companies": [
    { "name": "Waze", "type": "technology", "distance_m": 1200, "address": "..." }
  ],
  "processing_time": 2.8
}
```

### `GET /jobs/location/{city}`

Geocode a city and return nearby tech companies.

### `POST /jobs/match`

Match job listings against a provided resume for fit scoring.

---

## Applications

### `POST /applications`

Create a new job application record.

**Request Body** (`ApplicationCreateRequest`):
```json
{
  "company_name": "Google",
  "position_title": "Senior SWE",
  "job_url": "https://careers.google.com/...",
  "application_date": "2026-03-01",
  "notes": "Referred by Alice",
  "user_id": 1
}
```

**Response** `200 OK` (`ApplicationResponse`):
```json
{
  "id": 1,
  "company_name": "Google",
  "position_title": "Senior SWE",
  "job_url": "https://careers.google.com/...",
  "status": "applied",
  "application_date": "2026-03-01",
  "follow_up_date": null,
  "notes": "Referred by Alice",
  "created_at": "2026-03-01T10:00:00Z",
  "updated_at": "2026-03-01T10:00:00Z"
}
```

### `GET /applications`

List all applications for a user.

**Query Parameters**: `user_id` (default 1), `status` (optional filter)

### `PUT /applications/{application_id}`

Update application status, notes, or follow-up date.

**Request Body** (`ApplicationUpdateRequest`):
```json
{
  "status": "interviewing",
  "notes": "Phone screen scheduled for March 5",
  "follow_up_date": "2026-03-05"
}
```

**Status Workflow**:
```
applied → interviewing → offer
                ↓          ↓
            rejected    rejected
                ↓          ↓
            withdrawn   withdrawn
```

### `DELETE /applications/{application_id}`

Delete an application record.

### `GET /applications/follow-ups`

List applications with upcoming follow-up dates.

### `POST /applications/batch-update`

Async batch status update for multiple applications.

**Response** `202 Accepted`: `AsyncTaskResponse`

---

## User & Memory

### `GET /user/{user_id}/profile`

Get or create user profile.

### `GET /user/{user_id}/context`

Load full user context including profile, recent conversations, and summary.

**Response** `200 OK` (`UserContextResponse`):
```json
{
  "user_profile": { "skills": ["Python"], "experience_level": "senior", ... },
  "recent_conversations": [ { "intent": "resume_analysis", "summary": "...", ... } ],
  "context_summary": { "total_conversations": 15, "top_intents": ["resume_analysis"] },
  "history_count": 15
}
```

### `GET /user/{user_id}/insights`

Analyse conversation history and return patterns and recommendations.

**Response** `200 OK` (`ConversationAnalysisResponse`):
```json
{
  "insights": ["User frequently asks about resume formatting"],
  "patterns": ["Interview practice sessions increasing"],
  "recommendations": ["Consider focusing on system design preparation"],
  "conversation_count": 15,
  "agent_usage": { "resume": 8, "interview": 5, "knowledge": 2 }
}
```

### `POST /user/{user_id}/profile/update`

Manually update user profile fields.

---

## Unified Chat

### `POST /chat`

Natural-language interface that routes through the full LangGraph workflow.

**Request Body** (`ChatRequest`):
```json
{
  "message": "Can you review my resume for a senior backend role?",
  "user_id": 1,
  "session_id": null,
  "resume_text": "Jane Smith\nSenior Backend Engineer...",
  "job_description": "We are looking for...",
  "interview_role": null,
  "interview_level": null,
  "job_search_location": null,
  "interview_session_id": null,
  "interview_answer": null,
  "interview_question_id": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes (min 1 char) | Free-form user message |
| `user_id` | integer | No (default 1) | User identifier |
| `resume_text` | string | No | Attach resume for analysis |
| `job_description` | string | No | Target job posting |
| `interview_role` | string | No | Role for interview practice |
| `interview_level` | string | No | `junior`/`mid`/`senior`/`lead` |
| `job_search_location` | string | No | Location for job search |
| `interview_session_id` | string | No | Continue an existing interview |
| `interview_answer` | string | No | Answer to current question |
| `interview_question_id` | string | No | ID of question being answered |

**Response** `200 OK` (`ChatResponse`):
```json
{
  "response": "I've analyzed your resume. Here's what I found...",
  "intent": "resume_analysis",
  "confidence": 0.95,
  "agents_used": ["router", "resume", "memory"],
  "session_id": "conv-uuid-...",
  "processing_time": 5.2,
  "error": null,
  "data": {
    "overall_score": 7.5,
    "strengths": ["..."],
    "weaknesses": ["..."]
  },
  "interview_session_id": null
}
```

---

## Async Tasks

### `GET /tasks/{task_id}/status`

Check status of an asynchronous task.

**Response** `200 OK`:
```json
{
  "task_id": "abc123-...",
  "status": "completed",
  "result": { ... }
}
```

| Status | Meaning |
|--------|---------|
| `PENDING` | Queued |
| `STARTED` | In progress |
| `SUCCESS` | Completed |
| `FAILURE` | Failed |

### `GET /result/{task_id}`

Retrieve the result of a completed task.

---

## Error Responses

All endpoints return standard error structures:

```json
{
  "detail": "Error description"
}
```

| Code | Meaning |
|------|---------|
| `400` | Invalid request (validation failure) |
| `401` | Missing or invalid API key |
| `404` | Resource not found |
| `409` | Conflict (e.g., invalid status transition) |
| `422` | Validation error (Pydantic) |
| `429` | Rate limit exceeded |
| `500` | Internal server error |
