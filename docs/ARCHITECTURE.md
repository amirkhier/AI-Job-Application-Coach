# Architecture Guide

## System Overview

The AI Job Application Coach is a multi-agent system built on **FastAPI**, orchestrated by **LangGraph**, and backed by **GPT-4o-mini** via LangChain. The system provides resume analysis, interview practice, career knowledge Q\&A, job search, and application tracking through a unified REST API.

```
┌────────────────────────────────────────────────────────┐
│                  Client (HTTP / cURL)                   │
└────────────────────┬───────────────────────────────────┘
                     │  REST / JSON
┌────────────────────▼───────────────────────────────────┐
│              FastAPI Application (main.py)              │
│  ┌──────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │ CORS     │ │ API-Key Auth │ │ Rate Limit (60/m) │   │
│  └──────────┘ └──────────────┘ └───────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Input Validation Middleware            │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬───────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   LangGraph Workflow    │
        │   (StateGraph + nodes)  │
        └────────────┬────────────┘
                     │
   ┌─────────────────┼─────────────────────┐
   │                 │                     │
   ▼                 ▼                     ▼
┌────────┐    ┌────────────┐        ┌──────────┐
│ Memory │    │   Router   │        │ Summary  │
│ Agent  │    │   Agent    │        │  Node    │
└────┬───┘    └────┬───────┘        └──────────┘
     │             │ (intent routing)
     │    ┌────────┼────────┬────────────┐
     │    ▼        ▼        ▼            ▼
     │ ┌───────┐┌────────┐┌───────────┐┌──────────┐
     │ │Resume ││Intervw ││ Knowledge ││Job Search│
     │ │Agent  ││ Agent  ││   Agent   ││  Agent   │
     │ └───┬───┘└───┬────┘└─────┬─────┘└────┬─────┘
     │     │        │           │            │
     ▼     ▼        ▼           ▼            ▼
┌──────────────┐  ┌──────┐  ┌─────────┐  ┌──────────┐
│   MySQL DB   │  │OpenAI│  │ChromaDB │  │Nominatim │
│ (sessions +  │  │ API  │  │  (RAG)  │  │  (geo)   │
│ applications)│  └──────┘  └─────────┘  └──────────┘
└──────────────┘
```

---

## Component Details

### 1. API Layer (`app/main.py`)

The FastAPI application exposes **25 endpoints** grouped into six domains:

| Domain | Endpoints | Key Models |
|--------|-----------|-----------|
| Health | `GET /health` | `HealthResponse` |
| Resume | `POST /resume`, `/resume/improve`, `/resume/audit` | `ResumeRequest` → `ResumeResponse` |
| Interview | `POST /interview/start`, `/interview/answer`, `GET /interview/questions/{job_title}`, `POST /interview/report` | `InterviewStartRequest` → `InterviewStartResponse` |
| Knowledge | `POST /ask` | `KnowledgeQueryRequest` → `KnowledgeQueryResponse` |
| Jobs | `POST /jobs/search`, `/jobs/match`, `GET /jobs/location/{city}` | `JobSearchRequest` → `JobSearchResponse` |
| Applications | `POST /applications`, `GET /applications`, `PUT /applications/{id}`, `DELETE /applications/{id}`, `GET /applications/follow-ups`, `POST /applications/batch-update` | `ApplicationCreateRequest` → `ApplicationResponse` |
| User / Memory | `GET /user/{id}/profile`, `/user/{id}/context`, `/user/{id}/insights`, `POST /user/{id}/profile/update` | `UserContextResponse`, `ConversationAnalysisResponse` |
| Tasks | `GET /tasks/{id}/status`, `/result/{id}` | `AsyncTaskResponse` |
| Chat | `POST /chat` | `ChatRequest` → `ChatResponse` |

### 2. Middleware Stack (`app/middleware/`)

Requests pass through four middleware layers in order:

1. **RequestIdMiddleware** — Attaches a UUID correlation ID (`X-Request-ID` header) to every request for tracing.
2. **CORSMiddleware** — Configurable origin allowlist via `ALLOWED_ORIGINS`.
3. **ApiKeyMiddleware** (`auth.py`) — Optional API-key authentication. When `API_KEY` is set, all non-health endpoints require `X-API-Key` header.
4. **RateLimitMiddleware** (`rate_limit.py`) — Per-IP sliding-window rate limiter (default 60 req/min).
5. **InputValidationMiddleware** (`validation.py`) — Request body size and content validation.

### 3. Agent Architecture

All agents follow a consistent pattern:

```python
class Agent:
    def __init__(self, model="gpt-4o-mini", temperature=...):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._prompt = ChatPromptTemplate.from_messages([...])
        # Optional: self.chain = self._prompt | self.llm

    def public_method(self, ...):
        chain = self._prompt | self.llm   # or use self.chain
        result = chain.invoke({...})
        parsed = self._parse_llm_json(result.content)
        return parsed
```

#### Agent Inventory

| Agent | Module | Methods | LLM Temp | Description |
|-------|--------|---------|----------|-------------|
| **ResumeAgent** | `app/agents/resume.py` | `analyze_resume()`, `suggest_improvements()` | 0.1 | ATS-aware resume scoring, section feedback, keyword analysis, rewrite suggestions |
| **InterviewAgent** | `app/agents/interview.py` | `generate_questions()`, `evaluate_answer()`, `generate_session_summary()` | 0.3 | Role/level-specific questions, STAR-method evaluation, session summaries |
| **KnowledgeAgent** | `app/agents/knowledge.py` | `answer_question()`, `get_topic_summary()` | 0.2 | RAG-powered career Q\&A with source attribution |
| **JobSearchAgent** | `app/agents/job_search.py` | `search_jobs()`, `search_jobs_with_matching()` | 0.3 | LLM-generated listings with Nominatim geo-lookup, resume matching |
| **RouterAgent** | `app/agents/router.py` | `classify_intent()` | 0.0 | Intent classification (10 types) with keyword fallback |
| **MemoryAgent** | `app/agents/memory.py` | `load_user_context()`, `save_conversation_with_analysis()`, `update_profile_from_conversation()`, `get_conversation_insights()` | 0.2 | User profile persistence, conversation analysis, pattern detection |

### 4. LangGraph Workflow (`app/graph/workflow.py`)

The `JobCoachWorkflow` class builds a **LangGraph StateGraph** with 8 nodes:

```
START
  │
  ▼
memory_load  ── Load user profile & recent conversations
  │
  ▼
router       ── Classify intent via RouterAgent
  │
  ▼
[conditional] ── Branch based on intent
  │
  ├── resume_analysis / resume_improvement  →  resume_node
  ├── interview_practice / interview_start  →  interview_node
  ├── interview_answer                      →  interview_node
  ├── job_search                            →  job_search_node
  ├── career_advice / general_question      →  knowledge_node
  ├── application_tracking                  →  knowledge_node
  └── unknown                               →  knowledge_node
  │
  ▼
summary      ── Synthesise final response from specialist output
  │
  ▼
memory_save  ── Persist conversation + update user profile
  │
  ▼
END
```

**State Object** (`app/graph/state.py`):

```python
class JobCoachState(TypedDict):
    user_query: str
    user_id: int
    session_id: str
    intent: str
    confidence: float
    agents_used: list[str]
    response: str
    error_message: str
    processing_time: float
    # ... plus agent-specific fields
```

**Intent Types** (10 total):
`resume_analysis`, `resume_improvement`, `interview_practice`, `interview_start`, `interview_answer`, `job_search`, `career_advice`, `application_tracking`, `general_question`, `unknown`

### 5. RAG Pipeline (`app/rag/`)

The knowledge agent uses Retrieval-Augmented Generation:

1. **Corpus**: 4 Markdown guides in `app/rag/data/career_guides/`:
   - `resume_best_practices.md`
   - `interview_tips.md`
   - `salary_negotiation.md`
   - `industry_insights.md`

2. **Indexing** (`create_database.py`): Splits documents → embeds with OpenAI embeddings → stores in ChromaDB collection `career_guides`.

3. **Retrieval** (`query.py`): `query_knowledge_base(question, k=5)` performs similarity search → returns top-k chunks with scores. `get_formatted_context()` concatenates chunks for the LLM prompt.

### 6. Database Layer (`app/tools/database.py`)

`DatabaseManager` wraps MySQL via `mysql-connector-python`:

| Table | Purpose |
|-------|---------|
| `user_profiles` | User preferences, skills, experience level |
| `applications` | Job application tracking (status workflow: applied → interviewing → offer/rejected/withdrawn) |
| `conversations` | Conversation history with intent, summary, agent data |
| `interview_sessions` | Multi-turn interview state (questions, answers, scores) |

### 7. Async Task Processing (`app/tasks/`)

Celery tasks (Redis broker) for long-running operations:

| Task | Module | Trigger Endpoint |
|------|--------|-----------------|
| `full_resume_audit` | `resume_tasks.py` | `POST /resume/audit` |
| `generate_interview_report` | `interview_tasks.py` | `POST /interview/report` |
| `batch_application_update` | `application_tasks.py` | `POST /applications/batch-update` |

Task status is queryable via `GET /tasks/{task_id}/status` and results via `GET /result/{task_id}`.

---

## Configuration (`app/config.py`)

All settings are managed via Pydantic `BaseSettings` and sourced from environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `DEBUG` | `True` | Enable reload and verbose logging |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_FORMAT` | `text` | `json` for production, `text` for development |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Listen port |
| `API_KEY` | `None` | Set to require `X-API-Key` header |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `MYSQL_HOST` | `localhost` | MySQL hostname |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_USER` | `root` | Database user |
| `MYSQL_PASSWORD` | (empty) | Database password |
| `MYSQL_DATABASE` | `job_coach` | Database name |
| `OPENAI_API_KEY` | (empty) | Required for LLM calls |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma` | ChromaDB storage path |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |
| `USE_LANGGRAPH` | `False` | Feature flag for graph workflow |

---

## Directory Structure

```
AI-Job-Application-Coach/
├── app/
│   ├── main.py                 # FastAPI app + 25 endpoints
│   ├── config.py               # Pydantic settings
│   ├── logging_config.py       # Structured logging setup
│   ├── celery_worker.py        # Celery app factory
│   ├── agents/                 # 6 LLM-powered agents
│   │   ├── resume.py
│   │   ├── interview.py
│   │   ├── knowledge.py
│   │   ├── job_search.py
│   │   ├── router.py
│   │   └── memory.py
│   ├── graph/                  # LangGraph orchestration
│   │   ├── state.py            # JobCoachState + intent types
│   │   └── workflow.py         # StateGraph with 8 nodes
│   ├── middleware/             # Security + validation
│   │   ├── auth.py
│   │   ├── rate_limit.py
│   │   └── validation.py
│   ├── rag/                   # RAG pipeline
│   │   ├── create_database.py
│   │   ├── query.py
│   │   └── data/career_guides/
│   ├── tasks/                 # Celery async tasks
│   │   ├── resume_tasks.py
│   │   ├── interview_tasks.py
│   │   └── application_tasks.py
│   └── tools/                 # Database + agent tools
│       ├── database.py
│       ├── resume_tools.py
│       └── interview_tools.py
├── evaluation/                # Phase 5 evaluation framework
│   ├── scoring.py             # Automated scoring engine
│   ├── evaluator.py           # Agent evaluation runner
│   ├── reporters.py           # JSON/Markdown/Console reporters
│   ├── run.py                 # CLI entry point
│   ├── prompt_manager.py      # Prompt version management
│   ├── ab_testing.py          # A/B testing infrastructure
│   ├── statistics.py          # Statistical significance tests
│   ├── benchmarks.py          # Performance benchmarks
│   ├── datasets/              # 117 test cases across 5 agents
│   ├── prompts/               # Versioned prompt files
│   └── experiments/           # A/B experiment configs
├── tests/                     # 384 tests (pytest)
├── docs/                      # Project documentation
├── scripts/                   # Database setup SQL
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Data Flow: `/chat` Endpoint

The unified `/chat` endpoint demonstrates the complete data flow:

```
1. Client sends POST /chat { message, user_id, ... }
       │
2. RequestIdMiddleware assigns X-Request-ID
       │
3. Auth + Rate Limit + Validation middleware
       │
4. chat() handler extracts extra_kwargs
       │
5. JobCoachWorkflow.process_query()
       │
6. ┌── memory_load node ──────────────┐
   │  MemoryAgent.load_user_context() │
   └──────────────┬───────────────────┘
                  │
7. ┌── router node ──────────────────────────┐
   │  RouterAgent.classify_intent(query)     │
   │  → { intent, confidence, reasoning }    │
   └──────────────┬──────────────────────────┘
                  │
8. ┌── conditional routing ──────────────────┐
   │  intent → agent node mapping            │
   └──────────────┬──────────────────────────┘
                  │
9. ┌── specialist node (e.g. resume) ────────┐
   │  ResumeAgent.analyze_resume()           │
   │  → { score, strengths, weaknesses, ... }│
   └──────────────┬──────────────────────────┘
                  │
10.┌── summary node ─────────────────────────┐
   │  GPT-4o-mini synthesises user-facing    │
   │  response from specialist output         │
   └──────────────┬──────────────────────────┘
                  │
11.┌── memory_save node ─────────────────────┐
   │  MemoryAgent.save_conversation()        │
   │  MemoryAgent.update_profile()           │
   └──────────────┬──────────────────────────┘
                  │
12. ChatResponse returned to client
```
