# 💼 AI Job Application Coach

> A multi-agent AI system that acts as a personal career coach — powered by LangGraph orchestration, RAG-based knowledge retrieval, persistent memory, and asynchronous task processing.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Overview](#2-project-overview)
- [Key Features](#3-key-features)
- [Agent Architecture](#4-agent-architecture)
- [Tech Stack](#5-tech-stack)
- [System Diagram](#6-system-diagram-text-description)
- [Data Flow](#7-data-flow)
- [Development Phases](#8-milestones--development-phases)
- [Evaluation Metrics](#9-evaluation-metrics)
- [Risks & Limitations](#10-risks--limitations)
- [Future Improvements](#11-future-improvements)
- [Folder Structure](#12-folder-structure)
- [Architecture & Design Patterns](#13-architecture--design-patterns)
- [Quick Start Guide](#-quick-start-guide)
- [Troubleshooting](#-troubleshooting)

---

## Prerequisites

Before starting this project, ensure you have:
- Python 3.9 or higher
- OpenAI API key (for GPT-4o-mini and embeddings)
- MySQL database (local or cloud)
- RabbitMQ server (for Celery task queue)
- Basic understanding of:
  - LangChain and LangGraph
  - FastAPI and REST APIs
  - Vector databases (ChromaDB)
  - Asynchronous task processing

## 1. Project Title

**AI Job Application Coach with Multi-Agent Orchestration**

---

## 2. Project Overview

### Problem Statement
Job seekers face a fragmented and overwhelming process: tailoring resumes for each role, preparing for different interview styles, researching companies, and tracking dozens of applications simultaneously. Most candidates lack access to personalized career coaching, leading to generic applications and poor interview performance.

### Goal
Build an intelligent multi-agent system that acts as a personal career coach — helping users improve their resumes, practice interviews, discover relevant job opportunities, and track their application pipeline, all through natural language conversation with persistent memory across sessions.

## 3. Key Features

- **Resume Analysis & Feedback**: AI-powered resume review with actionable improvement suggestions
- **Mock Interview Practice**: Simulated interview sessions with role-specific questions and feedback
- **Job Discovery**: Search and recommend job listings from external APIs based on user profile
- **Career Knowledge Base**: RAG-powered advice on interview techniques, salary negotiation, and industry trends
- **Application Tracker**: Persistent database tracking applications, statuses, and follow-up reminders
- **Memory System**: Cross-session learning of user skills, preferences, and career goals
- **Async Report Generation**: Background processing for detailed resume audits and interview reports
- **REST API Interface**: FastAPI endpoints for integration with front-end applications

## 4. Agent Architecture

### 4.1 Agent List and Responsibilities

**1. Router Agent (Orchestrator)**
- Analyzes user input and classifies intent (resume help, interview prep, job search, application tracking)
- Delegates to appropriate specialized agents via conditional routing
- Manages multi-turn conversation flow
- **Tools**: None (uses LLM reasoning for intent classification)

**2. Resume Agent**
- Analyzes uploaded resume text for strengths and weaknesses
- Suggests improvements for specific job descriptions
- Generates tailored bullet points and summary statements
- **Tools**: `analyze_resume(resume_text: str, job_description: str) -> dict`

**3. Interview Agent**
- Generates role-specific interview questions (behavioral, technical, situational)
- Evaluates user answers and provides structured feedback
- Adapts difficulty based on user performance history
- **Tools**: `generate_questions(role: str, level: str) -> list`, `evaluate_answer(question: str, answer: str) -> dict`

**4. Job Search Agent**
- Searches for job listings via external APIs
- Filters results by location, role, experience level
- Finds nearby companies and coworking spaces using geolocation
- **Tools**: `search_jobs(query: str, location: str) -> list`, `get_city_center(city: str) -> tuple`, `find_nearby_offices(lat: float, lon: float) -> list`

**5. Knowledge Agent (RAG)**
- Retrieves career advice from a curated knowledge base (interview tips, salary guides, industry insights)
- Provides sourced answers with relevance scores
- **Tools**: `query_knowledge_base(query: str) -> str`

**6. Memory Agent**
- Maintains conversation history in MySQL database
- Tracks user profile: skills, target roles, application history
- Provides personalized context to other agents across sessions
- **Tools**: `save_conversation(session_id, text, summary)`, `load_history() -> list`, `update_profile(user_id, data) -> None`

### 4.2 Communication Flow

```
User Input
    ↓
Router Agent (classifies intent)
    ↓
    ├──→ "review my resume"     → Resume Agent
    ├──→ "practice interview"   → Interview Agent
    ├──→ "find jobs in Berlin"  → Job Search Agent
    ├──→ "how to negotiate?"    → Knowledge Agent (RAG)
    └──→ (always)               → Memory Agent (load context)
    ↓
Shared State (LangGraph State)
    ↓
    ├──→ Resume Agent may call Knowledge Agent for best-practice formatting
    ├──→ Interview Agent may call Memory Agent for past performance
    └──→ Job Search Agent may call Resume Agent for skill matching
    ↓
Memory Agent (persists conversation + updates profile)
    ↓
User Response
```

## 5. Tech Stack

### Core Framework
- **LangChain**: Agent creation, tool binding, prompt templates
- **LangGraph**: State graph orchestration, conditional routing, multi-agent workflows
- **OpenAI GPT-4o-mini**: Primary language model for all agents

### Data & Memory
- **MySQL**: Persistent user profiles, application tracker, conversation history
- **ChromaDB**: Vector database for career advice knowledge base
- **OpenAI Embeddings** (`text-embedding-3-small`): Document vectorization

### APIs & External Tools
- **OpenStreetMap Nominatim**: Geocoding for job location search
- **Overpass API**: Finding nearby offices, coworking spaces, company HQs
- **requests**: HTTP client for all external API calls

### Deployment & Orchestration
- **FastAPI**: REST API endpoints for all agent interactions
- **Celery**: Asynchronous task queue for report generation and batch processing
- **RabbitMQ**: Message broker for Celery workers

### Development Tools
- **python-dotenv**: Environment variable management
- **tiktoken**: Token counting for embeddings
- **unstructured**: Document loading for RAG pipeline

## 6. System Diagram (Text Description)

```
┌──────────────────────────────────────────────────────────────────────┐
│                       FastAPI REST Interface                         │
│  Endpoints: /resume, /interview, /jobs, /ask, /track, /result/{id}  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                          │
│                                                                      │
│  START → Router → [Resume/Interview/JobSearch/Knowledge] → Summary   │
│                                                                      │
│  State: {user_query, intent, resume_text, job_description,          │
│          interview_qa, job_results, knowledge_context,               │
│          user_profile, history, response}                            │
└───┬──────────────┬──────────────────┬──────────────────────────┘
    │              │                  │
    ▼              ▼                  ▼
┌────────┐  ┌──────────────┐  ┌──────────┐
│ChromaDB│  │OpenStreetMap │  │MySQL     │
│(RAG)   │  │APIs          │  │(Memory)  │
└────────┘  └──────────────┘  └──────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Celery Worker   │
        │  (Resume Audit / │
        │   Interview      │
        │   Reports)       │
        └──────────────────┘
```

## 7. Data Flow

### Request Flow (Synchronous)
1. User sends query via FastAPI endpoint (e.g., `POST /resume`)
2. Router Agent classifies intent → determines required agent(s)
3. Memory Agent loads user profile and conversation history from MySQL
4. LangGraph executes state transitions:
   - Resume Agent analyzes text and generates feedback
   - Interview Agent generates questions or evaluates answers
   - Job Search Agent queries external APIs for listings
   - Knowledge Agent retrieves advice from ChromaDB
5. Response Agent synthesizes final output
6. Memory Agent persists conversation summary and profile updates to MySQL
7. Response returned to user

### Background Flow (Asynchronous)
1. User requests detailed resume audit: `POST /resume/audit`
2. FastAPI submits Celery task → returns `task_id`
3. Celery worker runs multi-step analysis (resume parsing → skill extraction → gap analysis → recommendations)
4. User polls status: `GET /result/{task_id}`
5. When complete, detailed PDF-style report returned

### Mock Interview Flow
1. User starts session: `POST /interview/start` with target role
2. Interview Agent generates first question from LangGraph state
3. User submits answer: `POST /interview/answer`
4. Interview Agent evaluates answer, updates state, generates next question
5. After N rounds, Summary node generates performance report
6. Memory Agent persists weak areas for future practice

### RAG Pipeline (Offline)
1. Career guides, interview tips, and salary data placed in `data/career_guides/`
2. `create_database.py` loads and chunks documents
3. Embeddings generated via OpenAI API
4. Vectors stored in ChromaDB
5. Knowledge Agent queries at runtime with similarity search

## 8. Milestones & Development Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure and dependencies
- [ ] Configure MySQL schema: `users`, `conversations`, `applications` tables
- [ ] Implement basic LangGraph state machine with Router Agent
- [ ] Create FastAPI skeleton with health check and basic endpoints
- [ ] Set up environment variables and API keys

### Phase 2: Individual Agents (Week 2)
- [ ] Implement Resume Agent with multi-step analysis (parse → evaluate → suggest)
- [ ] Build Interview Agent with question generation and answer evaluation
- [ ] Create Knowledge Agent: load career guides into ChromaDB, implement RAG search
- [ ] Build Memory Agent with MySQL CRUD for profiles and conversations
- [ ] Implement Job Search Agent with Nominatim + Overpass integration

### Phase 3: Multi-Agent Orchestration (Week 3)
- [ ] Design shared state schema with all agent fields
- [ ] Implement conditional routing in Router Agent (intent classification)
- [ ] Wire cross-agent communication (Interview Agent reads Memory for weak areas)
- [ ] Build Summary node to synthesize multi-agent outputs
- [ ] Create multi-turn interview loop in LangGraph

### Phase 4: Deployment & Async Processing (Week 4)
- [ ] Set up RabbitMQ and Celery worker
- [ ] Implement async resume audit endpoint
- [ ] Build interview report generation as background task
- [ ] Create application tracker endpoints (CRUD)
- [ ] Add task polling and result retrieval

### Phase 5: Evaluation & Polish (Week 5)
- [ ] Implement evaluation metrics (routing accuracy, feedback quality)
- [ ] Test with 20+ sample queries across all agents
- [ ] Optimize prompts for resume analysis and interview feedback
- [ ] Add error handling, input validation, and retry logic
- [ ] Write documentation and record demo

## 9. Evaluation Metrics

### Functional Metrics
- **Router Accuracy**: % of queries routed to correct agent (target: >90%)
- **Resume Feedback Quality**: Human evaluation of suggestions (scale 1-5, target: >3.5)
- **Interview Question Relevance**: % of generated questions appropriate for the role
- **Job Search Precision**: % of returned listings matching user criteria
- **Memory Recall**: Correctly references past sessions in 5-query test

### Performance Metrics
- **End-to-End Latency**: Synchronous response time (target: <8s)
- **Async Task Completion**: Resume audit completes within 60s
- **Database Query Time**: MySQL and ChromaDB queries under 200ms
- **Agent Execution Time**: Per-agent processing breakdown

### Quality Metrics
- **RAG Relevance Score**: Average similarity score >0.7 for career advice retrieval
- **Interview Feedback Coherence**: LLM-as-Judge evaluation (scale 1-5)
- **Profile Learning**: Personalization improvement over 10 sessions
- **Application Tracking Accuracy**: 100% CRUD reliability

### Example Evaluation Suite
```python
test_queries = [
    "Review my resume for a Python developer position",
    "Give me 5 behavioral interview questions for a product manager role",
    "Find software engineering jobs in Tel Aviv",
    "How should I negotiate a salary offer?",           # RAG test
    "What were my weak areas from last interview?",     # Memory test
    "Track my application to Google — status: applied", # Tracker test
    "Generate a detailed resume audit report",          # Async test
]
```

## 10. Risks & Limitations

### Technical Risks
- **Resume Parsing Quality**: Unstructured text may confuse the LLM
  - *Mitigation*: Provide clear input format instructions, add preprocessing
- **Interview Evaluation Subjectivity**: LLM may give inconsistent scores
  - *Mitigation*: Use structured rubrics in prompts, calibrate with examples
- **API Rate Limits**: OpenStreetMap may throttle heavy usage
  - *Mitigation*: Cache geocoding results, rate-limit requests

### Functional Limitations
- **No Real Job API**: Uses OpenStreetMap for nearby companies, not live job boards
- **Text-Only Resumes**: No PDF/DOCX parsing (plain text input only)
- **Single Language**: English-only implementation
- **No Authentication**: Single-user focus, no login system

### Scalability Constraints
- **Sequential Agent Execution**: Agents run one at a time in LangGraph
- **MySQL Bottleneck**: Single database instance
- **Single Celery Worker**: No distributed processing

## 11. Future Improvements

### Short-Term Enhancements
- Add PDF/DOCX resume upload with file parsing
- Integrate real job board APIs (LinkedIn, Indeed) when available
- Add cover letter generation agent
- Implement user authentication and multi-user support
- Create web UI for interactive coaching sessions

### Medium-Term Enhancements
- Voice-based mock interviews using speech-to-text
- Video interview practice with body language tips
- LinkedIn profile optimization agent
- Email follow-up template generator
- Multi-language support for international job seekers

### Advanced Research Directions
- Fine-tuned resume scoring model based on hiring outcomes
- Agent self-improvement loop based on user satisfaction feedback
- Predictive job matching using embedding similarity
- Collaborative features (mentor review of AI-generated feedback)
- Portfolio/GitHub analysis agent

## 12. Folder Structure

```
capstone_job_coach/
│
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── celery_worker.py           # Celery configuration
│   ├── tasks.py                   # Async background tasks
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py              # Router Agent (intent classification)
│   │   ├── resume.py              # Resume Agent (analysis & feedback)
│   │   ├── interview.py           # Interview Agent (Q&A + evaluation)
│   │   ├── job_search.py          # Job Search Agent (external APIs)
│   │   ├── knowledge.py           # Knowledge Agent (RAG)
│   │   └── memory.py              # Memory Agent (MySQL persistence)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── resume_analyzer.py     # Resume parsing and scoring
│   │   ├── job_api.py             # Nominatim + Overpass integration
│   │   └── database.py            # MySQL CRUD utilities
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py               # LangGraph state schema
│   │   └── workflow.py            # State machine definition
│   │
│   └── rag/
│       ├── __init__.py
│       ├── create_database.py     # Vector DB setup
│       ├── query.py               # Similarity search
│       └── data/
│           └── career_guides/     # Markdown career advice documents
│               ├── interview_tips.md
│               ├── resume_best_practices.md
│               ├── salary_negotiation.md
│               └── industry_insights.md
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py
│   ├── test_graph.py
│   └── test_integration.py
│
├── evaluation/
│   ├── metrics.py                 # Evaluation framework
│   ├── benchmark_queries.json     # Test dataset
│   └── results/                   # Evaluation outputs
│
├── scripts/
│   ├── setup_db.sql               # MySQL schema (users, conversations, applications)
│   ├── populate_rag.py            # Load career guide documents
│   └── run.bat                    # Windows startup script
│
└── chroma/                        # ChromaDB storage (gitignored)
```

## 13. Architecture & Design Patterns

This project integrates several core agentic AI patterns and production-ready design principles:

### 13.1 State Graph Orchestration (LangGraph)
| Pattern | Implementation |
|---------|----------------|
| Typed State Management | `TypedDict` shared state across all agents |
| Conditional Routing | Router Agent classifies intent → dispatches to specialist |
| Sequential Node Chains | Resume Agent: Parse → Analyze → Suggest pipeline |
| Multi-Turn Loops | Interview Agent cycles through Q&A rounds with state accumulation |
| Cross-Agent Communication | Interview Agent reads Memory Agent data for personalization |

### 13.2 Tool-Augmented Agents (LangChain)
| Pattern | Implementation |
|---------|----------------|
| `@tool` Decorated Functions | All agent tools use LangChain's tool binding pattern |
| Direct API Integration | Job Search Agent calls Nominatim/Overpass via `requests` |
| Multi-Step LLM Reasoning | Resume Agent performs chained analysis with intermediate outputs |
| RAG Retrieval Tool | Knowledge Agent queries ChromaDB with similarity search |

### 13.3 Persistent Memory (MySQL)
| Pattern | Implementation |
|---------|----------------|
| Conversation History | Full session logs stored and summarized per interaction |
| User Profile Learning | Skills, preferences, and goals persist across sessions |
| Application Pipeline Tracking | CRUD operations for job application lifecycle |
| Context Injection | Memory Agent loads relevant history before each agent runs |

### 13.4 RAG Pipeline (ChromaDB + OpenAI Embeddings)
| Pattern | Implementation |
|---------|----------------|
| Document Ingestion | Markdown career guides loaded via `DirectoryLoader` |
| Text Chunking | `RecursiveCharacterTextSplitter` for optimal retrieval |
| Vector Storage | ChromaDB with `text-embedding-3-small` embeddings |
| Similarity Search | Knowledge Agent retrieves top-k relevant chunks |
| Prompt Grounding | LLM answers grounded in retrieved context with source attribution |

### 13.5 Deployment & Async Processing
| Pattern | Implementation |
|---------|----------------|
| REST API Layer | FastAPI with typed request/response models |
| Background Tasks | Celery + RabbitMQ for long-running resume audits |
| Task Polling | `GET /result/{task_id}` for async result retrieval |
| Service Separation | API server, worker, and broker run as independent processes |

---

## 🎯 Success Criteria

A successful implementation demonstrates:
- ✅ All 6 agents operational and communicating via LangGraph
- ✅ Router correctly classifies 5+ intent types
- ✅ Resume Agent provides actionable, structured feedback
- ✅ Interview Agent runs multi-turn Q&A sessions
- ✅ Knowledge Agent retrieves relevant career advice (RAG)
- ✅ Memory persists user profile and history across sessions
- ✅ At least one async endpoint (resume audit) via Celery
- ✅ Evaluation showing >85% routing accuracy on test queries
- ✅ Clean, modular, well-documented codebase

---

## 📋 Quick Start Guide

### 1. Environment Setup
```bash
mkdir capstone_job_coach && cd capstone_job_coach
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY=your_key_here
# MYSQL_HOST=localhost
# MYSQL_USER=root
# MYSQL_PASSWORD=your_password
# MYSQL_DATABASE=job_coach
```

### 3. Database Setup
```bash
mysql -u root -p < scripts/setup_db.sql
python app/rag/create_database.py
```

### 4. Start Services
```bash
# Terminal 1: RabbitMQ
rabbitmq-server

# Terminal 2: Celery worker
celery -A app.celery_worker worker --loglevel=info

# Terminal 3: FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test the System
```bash
# Resume review
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "5 years Python developer...", "job_description": "Senior Backend Engineer"}'

# Start mock interview
curl -X POST http://localhost:8000/interview/start \
  -H "Content-Type: application/json" \
  -d '{"role": "Product Manager", "level": "mid"}'

# Career advice (RAG)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I negotiate a higher salary?"}'
```

---

## 📦 Required Dependencies (requirements.txt)

```
# Core Framework
langchain==0.2.2
langchain-community==0.2.3
langchain-openai==0.1.8
langgraph>=0.0.20

# LLM & Embeddings
openai==1.31.1
tiktoken==0.7.0

# Vector Database
chromadb==0.5.0

# Database
mysql-connector-python==8.0.33

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Async Processing
celery==5.3.4

# External APIs
requests==2.31.0

# Document Processing
unstructured==0.14.4

# Utilities
python-dotenv==1.0.1
```

---

## 🔧 Troubleshooting

### Common Issues

**MySQL Connection Error**
- Ensure MySQL server is running
- Verify credentials in `.env` file
- Check database exists: `SHOW DATABASES;`

**ChromaDB Empty Results**
- Run `python app/rag/create_database.py` to populate the vector store
- Ensure `data/career_guides/` contains `.md` files

**Interview Session State Lost**
- Verify LangGraph state is being passed between turns
- Check MySQL for session persistence

**Celery Task Stuck**
- Ensure RabbitMQ is running: `rabbitmqctl status`
- Check worker logs: `celery -A app.celery_worker inspect active`

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👥 Contributors

| Name | Role | Contact |
|------|------|---------|
| _Your Name_ | Project Lead | _your.email@example.com_ |

## 📝 Changelog

| Date | Version | Description |
|------|---------|-------------|
| Feb 2026 | 1.0 | Initial project plan |
