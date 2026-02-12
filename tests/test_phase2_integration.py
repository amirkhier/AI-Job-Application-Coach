"""
Phase 2 Integration Test — tests all 5 agents via the FastAPI endpoints.
Requires the server running on http://127.0.0.1:8000
"""
import httpx
import json
import time
import sys

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0
client = httpx.Client(base_url=BASE, timeout=60.0)


def test(name: str, method: str, path: str, **kwargs):
    global PASS, FAIL
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  {method.upper()} {path}")
    try:
        if method == "post":
            r = client.post(path, **kwargs)
        else:
            r = client.get(path, **kwargs)

        status = r.status_code
        body = r.json()
        ok = 200 <= status < 300

        print(f"  STATUS: {status} {'✅' if ok else '❌'}")
        # Print a compact summary of the response
        if ok:
            PASS += 1
            print(f"  RESPONSE KEYS: {list(body.keys()) if isinstance(body, dict) else type(body).__name__}")
            # Print selected highlights
            for key in ["overall_score", "answer", "total_found", "session_id",
                        "history_count", "conversation_count", "jobs"]:
                if isinstance(body, dict) and key in body:
                    val = body[key]
                    if isinstance(val, list):
                        print(f"  {key}: [{len(val)} items]")
                    elif isinstance(val, str) and len(val) > 120:
                        print(f"  {key}: {val[:120]}...")
                    else:
                        print(f"  {key}: {val}")
        else:
            FAIL += 1
            detail = body.get("detail", body) if isinstance(body, dict) else body
            print(f"  ERROR: {str(detail)[:200]}")

        return body if ok else None

    except Exception as e:
        FAIL += 1
        print(f"  EXCEPTION: {e}")
        return None


# ================================================================
# 0. Health Check
# ================================================================
print("\n" + "🏥 HEALTH CHECK ".center(60, "="))
test("Health Check", "get", "/health")

# ================================================================
# 1. RESUME AGENT
# ================================================================
print("\n" + "📄 RESUME AGENT ".center(60, "="))

SAMPLE_RESUME = """
John Doe
Senior Software Engineer | john.doe@email.com | San Francisco, CA

SUMMARY
Experienced software engineer with 8 years building scalable web applications.
Proficient in Python, React, and cloud infrastructure.

EXPERIENCE
Senior Software Engineer — TechCorp Inc. (2020–Present)
- Led migration of monolith to microservices architecture serving 2M users
- Built real-time data pipeline processing 500K events per second using Kafka
- Mentored team of 5 junior engineers, improving code review throughput by 40%

Software Engineer — StartupXYZ (2017–2020)
- Developed REST APIs in Python/Flask handling 10K requests/minute
- Implemented CI/CD pipelines reducing deployment time from 2 hours to 15 minutes
- Designed PostgreSQL schema for multi-tenant SaaS platform

SKILLS
Python, JavaScript, React, AWS, Docker, Kubernetes, PostgreSQL, Redis, Kafka

EDUCATION
B.S. Computer Science — UC Berkeley (2016)
"""

JOB_DESC = """
Senior Backend Engineer — FinTech Startup
We're looking for a senior backend engineer to build our payment processing platform.
Requirements: Python, distributed systems, event-driven architecture, SQL databases.
Nice to have: Kafka, Kubernetes, financial domain experience.
"""

test("Resume Analysis", "post", "/resume",
     json={"resume_text": SAMPLE_RESUME, "job_description": JOB_DESC, "user_id": 1})

test("Resume Improvement", "post", "/resume/improve",
     json={"resume_text": SAMPLE_RESUME, "job_description": JOB_DESC, "user_id": 1})

# ================================================================
# 2. INTERVIEW AGENT
# ================================================================
print("\n" + "🎤 INTERVIEW AGENT ".center(60, "="))

start_result = test("Start Interview Session", "post", "/interview/start",
                     json={"role": "Python Backend Engineer", "level": "senior",
                           "question_count": 3, "user_id": 1})

if start_result:
    session_id = start_result.get("session_id", "")
    first_q = start_result.get("first_question", {})
    q_id = first_q.get("id", "q1")
    print(f"  SESSION: {session_id}")
    print(f"  FIRST Q: {first_q.get('question', 'N/A')[:100]}...")

    # Answer the first question
    test("Answer Interview Question", "post", "/interview/answer",
         json={
             "session_id": session_id,
             "question_id": q_id,
             "answer": "In my previous role at TechCorp, I led a project to migrate our monolithic application to microservices. I started by identifying bounded contexts, then created a strangler fig pattern to gradually extract services. We used Docker and Kubernetes for orchestration, implemented circuit breakers with resilience4j, and achieved 99.9% uptime during the transition. The result was 3x faster deployment cycles and better team autonomy."
         })

# Quick question generation (lightweight)
test("Quick Question Generation", "get", "/interview/questions/Data Engineer")

# ================================================================
# 3. KNOWLEDGE AGENT (RAG)
# ================================================================
print("\n" + "📚 KNOWLEDGE AGENT (RAG) ".center(60, "="))

test("Career Advice — Salary Negotiation", "post", "/ask",
     json={"query": "How should I negotiate my salary for a senior engineer role?", "user_id": 1})

test("Career Advice — Israeli Tech Market", "post", "/ask",
     json={"query": "What is the tech industry like in Israel? What are the salary ranges?", "user_id": 1})

# ================================================================
# 4. JOB SEARCH AGENT
# ================================================================
print("\n" + "🔍 JOB SEARCH AGENT ".center(60, "="))

test("Job Search — Tel Aviv", "post", "/jobs/search",
     json={"query": "Python Backend Engineer", "location": "Tel Aviv",
            "experience_level": "senior", "remote_ok": True, "count": 3, "user_id": 1})

test("City Location Info", "get", "/jobs/location/Berlin")

# ================================================================
# 5. MEMORY AGENT
# ================================================================
print("\n" + "🧠 MEMORY AGENT ".center(60, "="))

test("User Context", "get", "/user/1/context",
     params={"interaction_type": "general", "history_limit": 5})

test("Conversation Insights", "get", "/user/1/insights",
     params={"days_back": 7})

test("User Profile", "get", "/user/1/profile")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
print("=" * 60)

client.close()
sys.exit(0 if FAIL == 0 else 1)
