# Phase 4 – Deployment Roadmap

> **Status**: 8/30 checklist items complete (27%) — Docker infrastructure done  
> **Approach**: Incremental, test-as-you-go — each step produces a deployable, non-breaking state  
> **Branching strategy**: Feature branch per step (`phase4/step-N-description`), squash-merge into `main`

---

## Dependency Graph (Implementation Order)

```
Step 1  Production Config & Logging ──────────────────────────────┐
Step 2  Security Middleware ──────────────────────────────────────┤
Step 3  Application Tracker CRUD (complete) ─────────────────────┤ Independent
Step 4  Celery Infrastructure ───────────────────────────────────┤ cluster
Step 5  Async Resume Audit ──────────────── depends on Step 4 ──┤
Step 6  Background Report Generation ─────── depends on Step 4 ──┤
Step 7  Integration Testing & Hardening ──── depends on all ─────┘
```

Steps 1-3 are **independent** of each other and can be parallelised.  
Steps 5-6 strictly depend on Step 4 (Celery).  
Step 7 is the final gating step.

---

## Step 1 — Production Configuration & Structured Logging

### Goal
Replace ad-hoc `print()` / basic `logging` with structured, environment-aware configuration so every request is traceable in production.

### What exists today
- `logger = logging.getLogger(__name__)` in `main.py` — used inconsistently  
- CORS set to `allow_origins=["*"]` — not production-safe  
- No request-level tracing, no correlation IDs  
- `.env.example` / `.env.docker` with flat env vars, no env-specific profiles

### Deliverables

| File | Action |
|------|--------|
| `app/config.py` | **Create** — Pydantic `BaseSettings` class with env-specific profiles |
| `app/logging_config.py` | **Create** — `dictConfig`-based structured JSON logging |
| `app/main.py` | **Edit** — Wire up `config`, tighten CORS, add request-ID middleware |
| `.env.example` | **Edit** — Add new config keys |
| `docker-compose.yml` | **Edit** — Propagate new env vars |

### Implementation Details

#### 1a. `app/config.py` — Centralised Settings

```python
from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "text"       # json for prod, text for dev

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: str = "*"                          # comma-separated in prod
    API_KEY: Optional[str] = None                       # set in prod

    # Database
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "job_coach"

    # Redis / Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # ChromaDB
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIRECTORY: str = "./chroma"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Feature flags
    USE_LANGGRAPH: bool = False

    @property
    def cors_origins(self) -> list[str]:
        if self.ALLOWED_ORIGINS == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()
```

> **Why Pydantic `BaseSettings`?** Already a transitive dep via FastAPI. Single source of truth for every env var; type-coerced, validated at startup, IDE-friendly.  
> **Why not python-decouple / dynaconf?** Zero new deps; Pydantic is idiomatic in FastAPI projects.

#### 1b. `app/logging_config.py` — Structured Logging

```python
import logging
import logging.config
import uuid
from contextvars import ContextVar

# Per-request correlation ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "text": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
        },
    },
    "filters": {
        "request_id": {"()": "app.logging_config.RequestIdFilter"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "text",      # overridden at bootstrap depending on env
            "filters": ["request_id"],
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "loggers": {
        "uvicorn": {"level": "INFO"},
        "uvicorn.access": {"level": "WARNING"},
        "app": {"level": "DEBUG", "propagate": True},
    },
}

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get("-")
        return True

def setup_logging(log_level: str = "INFO", log_format: str = "text"):
    config = LOGGING_CONFIG.copy()
    config["root"]["level"] = log_level
    config["handlers"]["console"]["formatter"] = log_format
    logging.config.dictConfig(config)
```

> **Tooling note**: `python-json-logger` is a 20 KB pure-Python package — add to `requirements.txt`. For dev you keep human-readable text; in prod/Docker switch to JSON so that CloudWatch / ELK / Datadog can ingest structured logs natively.

#### 1c. Request-ID Middleware (in `main.py`)

```python
from starlette.middleware.base import BaseHTTPMiddleware
from app.logging_config import request_id_var

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
```

#### 1d. CORS — Environment-Aware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # ["*"] in dev, explicit list in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Acceptance Criteria
- [ ] `python -c "from app.config import settings; print(settings.ENVIRONMENT)"` works
- [ ] Every HTTP request gets an `X-Request-ID` header echoed back
- [ ] Log output in Docker includes the request ID
- [ ] `ALLOWED_ORIGINS=http://localhost:3000` restricts CORS correctly
- [ ] All existing 305 tests still pass

### Estimated Effort
~2-3 hours

---

## Step 2 — Security Middleware

### Goal
Add API-key authentication, rate limiting, input sanitisation, and SQL-injection protection so the service is defensible before going live.

### Deliverables

| File | Action |
|------|--------|
| `app/middleware/__init__.py` | **Create** |
| `app/middleware/auth.py` | **Create** — API key gate |
| `app/middleware/rate_limit.py` | **Create** — Token-bucket rate limiter |
| `app/middleware/validation.py` | **Create** — Input sanitisation |
| `app/main.py` | **Edit** — Register middleware stack |
| `requirements.txt` | **Edit** — Add `slowapi` (optional) |
| `tests/test_security.py` | **Create** |

### Implementation Details

#### 2a. API Key Authentication (`app/middleware/auth.py`)

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

# Paths that never require an API key
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if settings.API_KEY is None:                    # no key set → dev mode, skip
            return await call_next(request)
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return await call_next(request)
```

> **Best practice**: For a portfolio/demo project, a simple shared API key is sufficient. In a real SaaS, upgrade to OAuth2/JWT. The middleware is designed so that when `API_KEY` is unset, everything stays open — zero friction in development.

#### 2b. Rate Limiting (`app/middleware/rate_limit.py`)

**Option A — SlowAPI (preferred)**: Built on `limits` library, integrates with FastAPI natively.

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1",
)
```

Apply per-endpoint:
```python
@app.post("/resume")
@limiter.limit("10/minute")
async def analyze_resume(request: Request, ...):
    ...
```

**Option B — Manual in-memory** (if you want zero new deps):
```python
import time
from collections import defaultdict

class InMemoryRateLimiter:
    """Token-bucket rate limiter — good enough for single-instance deployments."""
    def __init__(self, rate: int = 60, per: int = 60):
        self.rate = rate
        self.per = per
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        bucket = self._buckets[key]
        bucket[:] = [t for t in bucket if now - t < self.per]
        if len(bucket) >= self.rate:
            return False
        bucket.append(now)
        return True
```

> **Recommendation**: Use SlowAPI with Redis backend. It handles distributed deployments (multiple app replicas) via shared Redis state, which you already have in your Docker Compose.

#### 2c. Input Sanitisation (`app/middleware/validation.py`)

```python
import re
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Block obvious SQL injection patterns in string payloads
SQL_INJECTION_PATTERNS = [
    r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\b.*\b(FROM|INTO|TABLE|SET)\b)",
    r"(--|;|/\*|\*/)",
    r"(\bOR\b\s+\d+\s*=\s*\d+)",
]
COMPILED = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]

MAX_BODY_SIZE = 1_000_000  # 1 MB

class InputValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Body size check
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            raise HTTPException(status_code=413, detail="Request body too large")

        # SQL injection check on JSON body (POST/PUT only)
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            body_text = body.decode("utf-8", errors="ignore")
            for pattern in COMPILED:
                if pattern.search(body_text):
                    raise HTTPException(status_code=400, detail="Potentially unsafe input detected")

            # Re-attach body for downstream consumption
            # (Starlette consumes the stream; use receive override)
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

        return await call_next(request)
```

> **Note**: You're already using parameterised queries in `DatabaseManager` (`.execute(query, params)`) which is the primary SQL injection defence. This middleware is a defence-in-depth layer — it catches attacks before they even reach your business logic.

#### 2d. Middleware Registration Order (in `main.py`)

Order matters — outermost middleware executes first:

```python
# 1. Request ID (always first — gives every log line a trace ID)
app.add_middleware(RequestIdMiddleware)
# 2. Input validation
app.add_middleware(InputValidationMiddleware)
# 3. API key auth
app.add_middleware(ApiKeyMiddleware)
# 4. Rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
# 5. CORS (must be last — Starlette processes middleware in reverse order)
app.add_middleware(CORSMiddleware, ...)
```

### Acceptance Criteria
- [ ] `X-API-Key` header enforced when `API_KEY` env var is set
- [ ] `/health`, `/docs` accessible without key
- [ ] Rapid-fire requests return 429 after threshold
- [ ] SQL injection payloads in resume_text return 400
- [ ] Request bodies > 1 MB rejected with 413
- [ ] All existing 305 tests pass (tests can skip auth by not setting `API_KEY`)

### Estimated Effort
~3-4 hours

---

## Step 3 — Complete Application Tracker CRUD

### Goal
Finish the partially-implemented application tracking endpoints: fix the `PUT` stub, add `DELETE`, add follow-up reminder logic, and add status workflow validation.

### What exists today
- `POST /applications` — functional, creates via `DatabaseManager.create_application()`
- `GET /applications` — functional, with optional `?status=` filter
- `PUT /applications/{id}` — partially stubbed: calls `db.update_application_status()` but returns hardcoded mock data
- `DELETE /applications/{id}` — **does not exist**
- `DatabaseManager` has `create_application()`, `get_applications()`, `update_application_status()` — no `get_application_by_id()` or `delete_application()`

### Deliverables

| File | Action |
|------|--------|
| `app/tools/database.py` | **Edit** — Add `get_application_by_id()`, `update_application()`, `delete_application()` |
| `app/main.py` | **Edit** — Fix `PUT`, add `DELETE`, add status workflow, add follow-up reminders |
| `app/main.py` | **Edit** — Add `ApplicationDeleteResponse` Pydantic model |
| `scripts/setup_db.sql` | **Verify** — Confirm `applications` table has all needed columns |
| `tests/test_application_tracker.py` | **Create** — Full CRUD + workflow tests |

### Implementation Details

#### 3a. Valid Status Transitions

```python
APPLICATION_STATUS_FLOW = {
    "applied":       ["screening", "rejected", "withdrawn"],
    "screening":     ["interviewing", "rejected", "withdrawn"],
    "interviewing":  ["offered", "rejected", "withdrawn"],
    "offered":       ["accepted", "rejected", "negotiating", "withdrawn"],
    "negotiating":   ["offered", "accepted", "rejected", "withdrawn"],
    "accepted":      [],
    "rejected":      [],
    "withdrawn":     [],
}
```

Enforce in the `PUT` handler:
```python
current_status = application["status"]
new_status = request.status
if new_status and new_status not in APPLICATION_STATUS_FLOW.get(current_status, []):
    raise HTTPException(
        status_code=422,
        detail=f"Cannot transition from '{current_status}' to '{new_status}'. "
               f"Allowed: {APPLICATION_STATUS_FLOW[current_status]}"
    )
```

#### 3b. Follow-Up Reminder Logic

Add a `GET /applications/follow-ups` endpoint:
```python
@app.get("/applications/follow-ups")
async def get_pending_follow_ups(user_id: int = 1, db = Depends(get_database)):
    """Return applications where follow_up_date <= today and status is active."""
    applications = db.get_applications(user_id=user_id)
    today = date.today()
    due = [
        a for a in applications
        if a.get("follow_up_date")
        and a["follow_up_date"] <= today
        and a["status"] not in ("accepted", "rejected", "withdrawn")
    ]
    return due
```

#### 3c. Database Methods to Add

```python
def get_application_by_id(self, application_id: int) -> Optional[Dict]:
    """Fetch a single application by primary key."""
    ...

def update_application(self, application_id: int, **fields) -> bool:
    """Update arbitrary fields on an application (status, notes, follow_up_date, job_url)."""
    ...

def delete_application(self, application_id: int) -> bool:
    """Hard-delete an application row."""
    ...
```

#### 3d. Fix `PUT` Endpoint

Replace the hardcoded mock response with a proper fetch-after-update:
```python
@app.put("/applications/{application_id}", response_model=ApplicationResponse)
async def update_application(application_id: int, request: ApplicationUpdateRequest, db = Depends(get_database)):
    existing = db.get_application_by_id(application_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Application not found")

    # Status workflow validation
    if request.status and request.status != existing["status"]:
        allowed = APPLICATION_STATUS_FLOW.get(existing["status"], [])
        if request.status not in allowed:
            raise HTTPException(status_code=422, detail=...)

    db.update_application(application_id, status=request.status, notes=request.notes,
                           follow_up_date=request.follow_up_date)
    updated = db.get_application_by_id(application_id)
    return ApplicationResponse(**updated)
```

### Acceptance Criteria
- [ ] `POST /applications` → 200, persisted in DB
- [ ] `GET /applications?status=interviewing` → filtered list
- [ ] `PUT /applications/1` with valid transition → 200
- [ ] `PUT /applications/1` with invalid transition → 422 with explanatory message
- [ ] `DELETE /applications/1` → 200, gone from DB
- [ ] `GET /applications/follow-ups` returns only overdue active items
- [ ] New tests cover happy paths + edge cases

### Estimated Effort
~2-3 hours

---

## Step 4 — Celery Infrastructure

### Goal
Stand up Celery with the Redis broker already running in Docker so that Steps 5 and 6 can dispatch long-running tasks.

### What exists today
- Redis container in `docker-compose.yml` — healthy  
- `celery_worker` service in `docker-compose.yml` — points to `app.celery_worker` which **does not exist yet**  
- `celery==5.3.4` and `redis==5.0.1` in `requirements.txt`  
- `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` env vars configured

### Deliverables

| File | Action |
|------|--------|
| `app/celery_worker.py` | **Create** — Celery app instance + autodiscovery |
| `app/tasks/__init__.py` | **Create** |
| `app/tasks/base.py` | **Create** — Base task class with retry / error handling |
| `app/main.py` | **Edit** — Expose `/tasks/{task_id}/status` endpoint |
| `docker-compose.yml` | **Verify** — Celery worker service already defined |
| `tests/test_celery.py` | **Create** — Worker connectivity + task lifecycle |

### Implementation Details

#### 4a. `app/celery_worker.py`

```python
from celery import Celery
from app.config import settings

celery_app = Celery(
    "job_coach",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                 # re-deliver if worker crashes
    worker_prefetch_multiplier=1,        # fair scheduling
    result_expires=86400,                # 24 h TTL on results
    task_soft_time_limit=300,            # 5 min soft limit
    task_time_limit=600,                 # 10 min hard kill
)

celery_app.autodiscover_tasks(["app.tasks"])
```

#### 4b. `app/tasks/base.py`

```python
from app.celery_worker import celery_app
from celery import Task
import logging

logger = logging.getLogger(__name__)

class BaseTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 120
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Task %s failed: %s", task_id, exc, exc_info=einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task %s completed successfully", task_id)
```

#### 4c. Task Status Endpoint

Replace the mock `/result/{task_id}` with a real Celery `AsyncResult` lookup:

```python
from celery.result import AsyncResult
from app.celery_worker import celery_app

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,       # PENDING | STARTED | SUCCESS | FAILURE | RETRY
        "result": None,
        "error": None,
    }
    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    return response
```

#### 4d. Verification — Local Smoke Test

```bash
# Terminal 1: Start Redis
docker compose up redis -d

# Terminal 2: Start Celery worker
celery -A app.celery_worker worker --loglevel=info

# Terminal 3: Send a test task from Python shell
from app.celery_worker import celery_app
result = celery_app.send_task("app.tasks.base.ping")   # add a trivial ping task for testing
print(result.get(timeout=10))
```

### Acceptance Criteria
- [ ] `celery -A app.celery_worker worker --loglevel=info` starts without error
- [ ] Worker connects to Redis broker and logs "ready"
- [ ] A trivial test task dispatched via `send_task()` returns a result
- [ ] `/tasks/{task_id}/status` returns real Celery state
- [ ] `docker compose --profile celery up` starts all 5 services
- [ ] Tests cover worker connectivity and task lifecycle

### Estimated Effort
~3-4 hours

---

## Step 5 — Async Resume Audit

### Goal
Implement the long-running resume audit as a Celery task — multi-step analysis that takes 30-60 seconds, too slow for a synchronous HTTP response.

### What exists today
- `POST /resume/audit` endpoint — returns a mock `AsyncTaskResponse` with a UUID  
- `ResumeAgent.analyze_resume()` — synchronous single-pass analysis  
- No Celery tasks defined yet (created in Step 4)

### Deliverables

| File | Action |
|------|--------|
| `app/tasks/resume_tasks.py` | **Create** — `detailed_resume_audit` Celery task |
| `app/main.py` | **Edit** — Wire `/resume/audit` to dispatch real Celery task |
| `tests/test_async_resume.py` | **Create** |

### Implementation Details

#### 5a. `app/tasks/resume_tasks.py`

```python
from app.celery_worker import celery_app
from app.tasks.base import BaseTaskWithRetry
from app.agents.resume import ResumeAgent
import logging

logger = logging.getLogger(__name__)

@celery_app.task(base=BaseTaskWithRetry, bind=True, name="resume.detailed_audit")
def detailed_resume_audit(self, resume_text: str, job_description: str = None, user_id: int = 1):
    """
    Multi-step resume audit:
      1. Basic analysis (score, strengths, weaknesses)
      2. ATS compatibility deep-scan
      3. Skill extraction & gap analysis (vs. job description)
      4. Section-by-section rewrite suggestions
      5. Final comprehensive report assembly
    """
    agent = ResumeAgent()

    # Progress updates via Celery meta
    self.update_state(state="PROGRESS", meta={"step": 1, "total": 5, "detail": "Basic analysis"})
    analysis = agent.analyze_resume(resume_text, job_description)

    self.update_state(state="PROGRESS", meta={"step": 2, "total": 5, "detail": "Improvement suggestions"})
    improvements = agent.suggest_improvements(resume_text, job_description)

    self.update_state(state="PROGRESS", meta={"step": 3, "total": 5, "detail": "Skill gap analysis"})
    # ... skill extraction pass ...

    self.update_state(state="PROGRESS", meta={"step": 4, "total": 5, "detail": "Section rewrites"})
    # ... section-by-section rewrite ...

    self.update_state(state="PROGRESS", meta={"step": 5, "total": 5, "detail": "Report assembly"})
    report = {
        "analysis": analysis,
        "improvements": improvements,
        "audit_complete": True,
    }
    return report
```

#### 5b. Wire the Endpoint

```python
@app.post("/resume/audit", response_model=AsyncTaskResponse, status_code=202)
async def request_detailed_resume_audit(request: ResumeRequest):
    from app.tasks.resume_tasks import detailed_resume_audit
    task = detailed_resume_audit.delay(
        resume_text=request.resume_text,
        job_description=request.job_description,
        user_id=request.user_id,
    )
    return AsyncTaskResponse(
        task_id=task.id,
        status="queued",
        message="Resume audit task queued for processing",
        estimated_completion=None,
    )
```

#### 5c. Progress Polling

Extend `/tasks/{task_id}/status` (from Step 4) to include progress metadata:

```python
if result.status == "PROGRESS":
    response["progress"] = result.info   # {"step": 2, "total": 5, "detail": "..."}
```

### Acceptance Criteria
- [ ] `POST /resume/audit` returns 202 with a real task ID
- [ ] Polling `/tasks/{id}/status` shows PENDING → STARTED → PROGRESS → SUCCESS
- [ ] Progress metadata includes step number and description
- [ ] Completed task result contains full analysis + improvements
- [ ] Worker retries on transient OpenAI failures (up to 3 times)
- [ ] Tests mock the LLM and validate task lifecycle

### Estimated Effort
~3-4 hours

---

## Step 6 — Background Report Generation

### Goal
Add Celery tasks for interview performance reports and batch application tracking updates.

### Deliverables

| File | Action |
|------|--------|
| `app/tasks/interview_tasks.py` | **Create** — `generate_interview_report` task |
| `app/tasks/application_tasks.py` | **Create** — `batch_status_update`, `generate_follow_up_reminders` tasks |
| `app/main.py` | **Edit** — Add `/interview/report` and `/applications/batch-update` endpoints |
| `tests/test_async_tasks.py` | **Create** |

### Implementation Details

#### 6a. Interview Performance Report

```python
@celery_app.task(base=BaseTaskWithRetry, bind=True, name="interview.performance_report")
def generate_interview_report(self, user_id: int, session_id: str):
    """
    Generates a comprehensive interview performance report:
      1. Aggregate all Q&A pairs from the session
      2. Per-question scoring breakdown
      3. Pattern analysis (strengths across questions)
      4. Personalised improvement plan
      5. Recommended practice questions for weak areas
    """
    ...
```

#### 6b. Application Batch Update

```python
@celery_app.task(name="applications.batch_status_check")
def batch_status_check(user_id: int):
    """
    For all active applications:
      - Check if follow_up_date is overdue
      - Generate reminder notifications
      - Flag stale applications (no activity > 30 days)
    """
    ...
```

#### 6c. Report Template System

```python
REPORT_TEMPLATES = {
    "interview_summary": {
        "sections": ["overview", "per_question_breakdown", "strength_analysis",
                      "improvement_areas", "recommended_practice"],
        "format": "markdown",
    },
    "resume_audit": {
        "sections": ["executive_summary", "ats_analysis", "content_analysis",
                      "skill_gaps", "rewrite_suggestions"],
        "format": "markdown",
    },
}
```

### Acceptance Criteria
- [ ] `POST /interview/report` dispatches report generation, returns 202
- [ ] Report result contains all template sections
- [ ] Batch application check identifies stale/overdue items
- [ ] Tasks are discoverable by Celery worker via autodiscover

### Estimated Effort
~3-4 hours

---

## Step 7 — Integration Testing & Hardening

### Goal
Validate all Phase 4 work end-to-end, harden error handling, and update documentation.

### Deliverables

| File | Action |
|------|--------|
| `tests/test_phase4_integration.py` | **Create** — Cross-cutting E2E tests |
| `app/main.py` | **Edit** — Global exception handler, graceful degradation |
| `docs/IMPLEMENTATION_CHECKLIST.md` | **Edit** — Mark Phase 4 items complete |
| `README.md` | **Edit** — Update setup instructions for Celery, env vars |
| `.env.example` | **Edit** — Add all new env vars |

### Implementation Details

#### 7a. Global Exception Handler

```python
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id_var.get("-")},
    )
```

#### 7b. Graceful Degradation

When Celery/Redis is unavailable, async endpoints should fall back to synchronous execution instead of crashing:

```python
@app.post("/resume/audit", status_code=202)
async def request_detailed_resume_audit(request: ResumeRequest):
    try:
        from app.tasks.resume_tasks import detailed_resume_audit
        task = detailed_resume_audit.delay(...)
        return AsyncTaskResponse(task_id=task.id, status="queued", ...)
    except Exception as e:
        logger.warning("Celery unavailable, falling back to sync: %s", e)
        # Run synchronously as fallback
        result = ResumeAgent().analyze_resume(request.resume_text, request.job_description)
        return {"task_id": "sync", "status": "completed", "result": result}
```

#### 7c. Health Check Enhancement

Extend `/health` to report subsystem status:

```python
@app.get("/health")
async def health_check():
    checks = {
        "database": _check_db(),
        "redis": _check_redis(),
        "celery": _check_celery_worker(),
    }
    overall = "healthy" if all(checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "timestamp": datetime.now().isoformat()}
```

#### 7d. Integration Test Matrix

| Test Scenario | Validates |
|---|---|
| Full resume audit → poll → result | Celery E2E |
| CRUD application → status transition → follow-up | Tracker workflow |
| API key required in prod mode | Auth middleware |
| Rate limit triggered at threshold | Rate limiter |
| SQL injection payload rejected | Input validation |
| Redis down → sync fallback | Graceful degradation |
| Concurrent requests with unique IDs | Request tracing |

### Acceptance Criteria
- [ ] All 305 existing tests still pass
- [ ] New Phase 4 tests add ≥50 test cases
- [ ] `/health` reports subsystem statuses
- [ ] Async endpoints gracefully degrade when Celery is unavailable
- [ ] README updated with Phase 4 setup instructions
- [ ] Checklist updated: Phase 4 → 30/30

### Estimated Effort
~3-4 hours

---

## Summary — Effort & Sequencing

| Step | Description | Depends On | Est. Hours | Files Changed/Created |
|------|-------------|------------|-----------|----------------------|
| 1 | Production Config & Logging | — | 2-3 h | 4 created, 3 edited |
| 2 | Security Middleware | Step 1 (config) | 3-4 h | 5 created, 2 edited |
| 3 | Application Tracker CRUD | — | 2-3 h | 1 created, 2 edited |
| 4 | Celery Infrastructure | Step 1 (config) | 3-4 h | 4 created, 2 edited |
| 5 | Async Resume Audit | Step 4 | 3-4 h | 2 created, 1 edited |
| 6 | Background Reports | Step 4 | 3-4 h | 3 created, 1 edited |
| 7 | Integration & Hardening | All | 3-4 h | 2 created, 4 edited |
| **Total** | | | **~20-26 h** | **~21 created, ~15 edited** |

### Recommended Execution Order

```
Week 4, Day 1   →  Step 1 (config/logging)
Week 4, Day 1-2 →  Step 2 (security) + Step 3 (CRUD) in parallel
Week 4, Day 2-3 →  Step 4 (Celery infra)
Week 4, Day 3-4 →  Step 5 (async resume) + Step 6 (reports)
Week 4, Day 4-5 →  Step 7 (integration + hardening)
```

### Key Best Practices Embedded

1. **Zero-downtime progression** — Every step leaves the app in a deployable state  
2. **Feature flags over big-bang** — `API_KEY=None` bypasses auth in dev; `USE_LANGGRAPH` already exists  
3. **Defence in depth** — Parameterised queries (exists) + input validation middleware + rate limiting  
4. **Graceful degradation** — Celery unavailable → sync fallback, not a 500 error  
5. **Observability first** — Request IDs and structured logging before adding complexity  
6. **Test parity** — Every step includes its own test suite; all 305 existing tests must remain green  
7. **No new heavy deps** — `pydantic-settings`, `python-json-logger`, `slowapi` are lightweight; everything else is already in `requirements.txt`

### New Dependencies to Add

```
pydantic-settings>=2.0     # BaseSettings (separated from pydantic v2)
python-json-logger>=2.0    # Structured JSON log output
slowapi>=0.1.9             # Rate limiting for FastAPI
```
