# User Guide

## What Is the AI Job Application Coach?

A multi-agent AI system that acts as your personal career coach. It can:

- **Analyse your resume** — score, identify strengths/weaknesses, check ATS compatibility
- **Improve your resume** — rewrite bullets, suggest additions, prioritise changes
- **Run mock interviews** — generate role-specific questions, evaluate your answers with STAR-method feedback
- **Answer career questions** — salary negotiation, interview tips, industry insights (grounded in a curated knowledge base)
- **Search for jobs** — generate relevant listings with location matching
- **Track your applications** — record, update status, manage follow-ups

---

## Getting Started

### 1. Start the Server

```bash
# Docker (recommended)
docker-compose up -d

# Or manually
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Open the API Docs

Navigate to **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Using the Chat Interface

The easiest way to use the system is through the unified `/chat` endpoint, which auto-routes your message to the right agent.

### Resume Analysis

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Please review my resume",
    "resume_text": "Jane Smith\nSenior Backend Engineer\n..."
  }'
```

The response includes a structured analysis with score, strengths, weaknesses, and ATS feedback.

### Resume Improvement

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me improve my resume for this job",
    "resume_text": "...",
    "job_description": "Senior Backend Engineer at FinTech..."
  }'
```

### Interview Practice

Start a mock interview:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to practice for a senior backend engineer interview",
    "interview_role": "Backend Engineer",
    "interview_level": "senior"
  }'
```

The response includes the first question and an `interview_session_id`. Continue the interview by sending answers:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Here is my answer",
    "interview_session_id": "<session-id-from-previous-response>",
    "interview_answer": "At my previous company, I was responsible for...",
    "interview_question_id": "q1"
  }'
```

Each answer gets STAR-method evaluation with scores and feedback. After all questions are answered, you receive a complete session summary.

### Career Questions

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How should I negotiate my salary after receiving an offer?"
  }'
```

Answers are grounded in the built-in career guide library covering salary negotiation, interview preparation, resume best practices, and industry insights.

### Job Search

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find me Python developer jobs in Tel Aviv",
    "job_search_location": "Tel Aviv"
  }'
```

---

## Using Structured Endpoints

For programmatic integration, use the domain-specific endpoints directly:

### Resume

```bash
# Analyse
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "job_description": "..."}'

# Improve
curl -X POST http://localhost:8000/resume/improve \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "...", "job_description": "..."}'
```

### Interview

```bash
# Start session
curl -X POST http://localhost:8000/interview/start \
  -H "Content-Type: application/json" \
  -d '{"role": "Backend Engineer", "level": "senior", "question_count": 5}'

# Answer question
curl -X POST http://localhost:8000/interview/answer \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "question_id": "q1", "answer": "..."}'

# Quick questions (no session)
curl http://localhost:8000/interview/questions/Backend%20Engineer?level=senior&count=3
```

### Knowledge

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best practices for writing a technical resume?"}'
```

### Job Search

```bash
curl -X POST http://localhost:8000/jobs/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer", "location": "Tel Aviv", "experience_level": "senior", "count": 5}'
```

### Application Tracking

```bash
# Create
curl -X POST http://localhost:8000/applications \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Google", "position_title": "Senior SWE", "job_url": "https://..."}'

# List
curl http://localhost:8000/applications?user_id=1

# Update status
curl -X PUT http://localhost:8000/applications/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "interviewing", "notes": "Phone screen scheduled"}'

# Check follow-ups
curl http://localhost:8000/applications/follow-ups?user_id=1

# Delete
curl -X DELETE http://localhost:8000/applications/1
```

---

## Application Status Workflow

Applications follow a defined status progression:

```
applied → interviewing → offer
    ↓          ↓           ↓
 rejected   rejected    rejected
    ↓          ↓           ↓
 withdrawn  withdrawn   withdrawn
```

Only valid transitions are allowed. For example, an application in `applied` status can move to `interviewing`, `rejected`, or `withdrawn`, but not directly to `offer`.

---

## Memory & User Context

The system remembers your interactions. View your profile and conversation history:

```bash
# View profile
curl http://localhost:8000/user/1/profile

# Full context (profile + recent conversations + summary)
curl http://localhost:8000/user/1/context

# AI-generated insights from your conversation patterns
curl http://localhost:8000/user/1/insights

# Update profile manually
curl -X POST http://localhost:8000/user/1/profile/update \
  -H "Content-Type: application/json" \
  -d '{"skills": ["Python", "AWS"], "experience_level": "senior"}'
```

---

## Tips

1. **Provide a job description** when analysing your resume for tailored keyword analysis and scoring.
2. **Use the chat endpoint** for conversational interactions — it automatically routes to the right agent.
3. **Use structured endpoints** for programmatic integrations and automation.
4. **Track follow-ups** — set `follow_up_date` when updating applications to get reminders.
5. The system learns from your interactions — the more you use it, the more personalised the advice becomes.
