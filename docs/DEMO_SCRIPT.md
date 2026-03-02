# Demo Script

Step-by-step walkthrough demonstrating all major features of the AI Job Application Coach.

**Prerequisites**: Server running on `http://localhost:8000` (see [Deployment Runbook](DEPLOYMENT_RUNBOOK.md)).

---

## Step 1 — Health Check

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

**Expected**: `"status": "healthy"` (or `"degraded"` if Redis/ChromaDB are unavailable).

---

## Step 2 — Resume Analysis

Analyse a sample resume against a target job description:

```bash
curl -s -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jane Smith\nSenior Backend Engineer | jane@email.com | Tel Aviv\n\nEXPERIENCE\nSenior Backend Engineer — CyberTech Ltd. (2021-Present)\n- Architected event-driven microservices processing 1M events/day using Kafka\n- Led migration from AWS to hybrid cloud, reducing costs by 35%\n- Mentored team of 4 junior developers\n\nBackend Developer — DataFlow Startup (2018-2021)\n- Built REST APIs in Python/FastAPI handling 50K concurrent users\n- Implemented CI/CD pipeline cutting deployment time by 70%\n\nSKILLS\nPython, Go, FastAPI, AWS, Docker, Kubernetes, PostgreSQL, Redis, Kafka\n\nEDUCATION\nB.Sc. Computer Science — Technion (2016)",
    "job_description": "Senior Backend Engineer — FinTech. Requirements: 5+ years Python, distributed systems, AWS, payment processing."
  }' | python -m json.tool
```

**What to show**:
- `overall_score` (1–10)
- `strengths` and `weaknesses` arrays
- `ats_compatibility.score` and `missing_keywords`
- `section_feedback` per resume section

---

## Step 3 — Resume Improvement

Request concrete rewrite suggestions:

```bash
curl -s -X POST http://localhost:8000/resume/improve \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jane Smith\nSenior Backend Engineer | jane@email.com | Tel Aviv\n\nEXPERIENCE\nSenior Backend Engineer — CyberTech Ltd. (2021-Present)\n- Architected event-driven microservices processing 1M events/day using Kafka\n- Led migration from AWS to hybrid cloud, reducing costs by 35%\n\nSKILLS\nPython, Go, FastAPI, AWS, Docker, Kubernetes\n\nEDUCATION\nB.Sc. Computer Science — Technion (2016)",
    "job_description": "Senior Backend Engineer — FinTech. 5+ years Python, distributed systems."
  }' | python -m json.tool
```

**What to show**:
- `improved_summary` — AI-generated professional summary
- `improved_bullets[].original` vs `improved_bullets[].improved` — side-by-side comparison
- `priority_actions` — ordered list of highest-impact changes

---

## Step 4 — Mock Interview

### Start an interview session:

```bash
curl -s -X POST http://localhost:8000/interview/start \
  -H "Content-Type: application/json" \
  -d '{
    "role": "Backend Engineer",
    "level": "senior",
    "question_count": 3
  }' | python -m json.tool
```

**Save** the `session_id` from the response. **Show** the `first_question`.

### Answer a question:

```bash
curl -s -X POST http://localhost:8000/interview/answer \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<paste-session-id>",
    "question_id": "q1",
    "answer": "At CyberTech, I led our migration from a monolithic architecture to microservices. The system was handling 500K events/day and we needed to scale to 1M. I designed the Kafka-based event pipeline, coordinated with 3 teams over 6 months, and we achieved zero downtime during the transition. The result was a 60% improvement in deployment speed and the ability to scale individual services independently."
  }' | python -m json.tool
```

**What to show**:
- `feedback.overall_score` — STAR-method score
- `feedback.strength_areas` — what the candidate did well
- `feedback.improvement_areas` — what to work on
- `next_question` — the next question in the session
- After the final question: `session_summary` with aggregate performance

---

## Step 5 — Career Knowledge Q&A

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best strategies for negotiating a salary increase?"
  }' | python -m json.tool
```

**What to show**:
- `answer` — detailed, grounded response
- `sources` — which career guide documents were used (RAG attribution)
- `relevance_score` — confidence in the answer
- `related_topics` — suggested follow-up questions

---

## Step 6 — Job Search

```bash
curl -s -X POST http://localhost:8000/jobs/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python backend developer",
    "location": "Tel Aviv",
    "experience_level": "senior",
    "remote_ok": true,
    "count": 3
  }' | python -m json.tool
```

**What to show**:
- `jobs[]` — generated listings with company, description, salary, skills
- `location_info` — geocoded location data
- `nearby_companies` — real companies near the search location

---

## Step 7 — Application Tracking

### Create an application:

```bash
curl -s -X POST http://localhost:8000/applications \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Google",
    "position_title": "Senior Backend Engineer",
    "job_url": "https://careers.google.com/jobs/12345",
    "notes": "Referred by Alice from the platform team"
  }' | python -m json.tool
```

### Update application status:

```bash
curl -s -X PUT http://localhost:8000/applications/1 \
  -H "Content-Type: application/json" \
  -d '{
    "status": "interviewing",
    "notes": "Phone screen scheduled for March 10",
    "follow_up_date": "2026-03-10"
  }' | python -m json.tool
```

### List all applications:

```bash
curl -s http://localhost:8000/applications?user_id=1 | python -m json.tool
```

---

## Step 8 — Unified Chat (Full Workflow)

Demonstrate the LangGraph-powered chat that routes through memory → router → specialist → summary → memory:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you review my resume and suggest improvements for a senior backend role at a fintech company?",
    "resume_text": "Jane Smith\nSenior Backend Engineer | jane@email.com\n\nEXPERIENCE\nSenior Backend Engineer — CyberTech (2021-Present)\n- Built microservices with Kafka\n- Reduced infrastructure costs by 35%\n\nSKILLS\nPython, AWS, Docker, Kubernetes",
    "job_description": "Senior Backend Engineer at PayTech. 5+ years, Python, distributed systems, payment processing."
  }' | python -m json.tool
```

**What to show**:
- `intent` — how the router classified the message
- `confidence` — router certainty
- `agents_used` — which agents participated (e.g., `["router", "resume", "memory"]`)
- `response` — synthesised natural-language answer
- `data` — structured agent output (scores, analysis)
- `processing_time` — end-to-end latency

---

## Step 9 — Evaluation Framework (Developer Demo)

Show the evaluation tooling built in Phase 5:

```bash
# Dry-run evaluation across all 117 test cases
python -m evaluation.run --dry-run --verbose

# Run prompt regression tests
python -m pytest tests/test_prompt_regression.py -v

# Run performance benchmarks (mock mode)
python -m evaluation.benchmarks --mock --iterations 5

# View generated reports
cat evaluation/reports/benchmarks.md
```

---

## Summary of Capabilities

| Feature | Endpoint | Agent |
|---------|----------|-------|
| Resume Analysis | `POST /resume` or `/chat` | ResumeAgent |
| Resume Improvement | `POST /resume/improve` or `/chat` | ResumeAgent |
| Mock Interview | `POST /interview/start` + `/interview/answer` or `/chat` | InterviewAgent |
| Career Q&A | `POST /ask` or `/chat` | KnowledgeAgent (RAG) |
| Job Search | `POST /jobs/search` or `/chat` | JobSearchAgent |
| Application Tracking | `POST/GET/PUT/DELETE /applications` | Database |
| User Memory | `GET /user/{id}/context` | MemoryAgent |
| Async Tasks | `POST /resume/audit`, `/interview/report` | Celery |
| Unified Chat | `POST /chat` | Full LangGraph workflow |
