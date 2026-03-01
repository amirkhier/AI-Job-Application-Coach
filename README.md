# 💼 AI Job Application Coach

> A multi-agent AI system that acts as a personal career coach — powered by LangGraph orchestration, RAG-based knowledge retrieval, persistent memory, and asynchronous task processing.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://python.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended) 🐳

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running
- OpenAI API key

**Easy Setup:**
```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Job-Application-Coach

# Copy environment template and add your OpenAI API key
cp .env.docker .env
# Edit .env and replace 'your_openai_api_key_here' with your actual API key

# Start all services with one command
./docker-setup.sh          # Linux/macOS
# OR
docker-setup.bat           # Windows
```

**What Docker provides:**
- ✅ **No MySQL installation needed** - MySQL 8.0 in container
- ✅ **No Redis setup required** - Redis for Celery (future use)
- ✅ **ChromaDB included** - Vector database for RAG
- ✅ **Automatic service orchestration** - All dependencies managed
- ✅ **Health checks** - Ensures all services are ready
- ✅ **Data persistence** - Database and vector data preserved

**Access your application:**
- 🚀 **Main API**: http://localhost:8000
- 📚 **Interactive Docs**: http://localhost:8000/docs  
- 🔍 **Health Check**: http://localhost:8000/health
- �️ **MySQL Database**: localhost:3307 (external port to avoid conflicts)
- 📊 **ChromaDB**: http://localhost:8001
- 💾 **Redis**: localhost:6380 (external port to avoid conflicts)

### Option 2: Manual Setup

**Prerequisites:**
- Python 3.9 or higher
- MySQL Server (local or cloud)
- OpenAI API key
- Git

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Job-Application-Coach

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials:
# OPENAI_API_KEY=your_openai_api_key_here
# MYSQL_HOST=localhost
# MYSQL_USER=root
# MYSQL_PASSWORD=your_mysql_password
# MYSQL_DATABASE=job_coach
```

### 3. Database Setup

```bash
# Create database and tables
mysql -u root -p < scripts/setup_db.sql
```

### 4. Start the Application

```bash
# Run the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` to see the interactive API documentation.

---

## 🐳 Docker Management

### Basic Commands
```bash
# Start all services
docker-compose up -d

# Stop all services  
docker-compose down

# View logs
docker-compose logs -f app

# Restart application
docker-compose restart app

# Database shell access
docker-compose exec mysql mysql -u jobcoach -p job_coach

# Application shell access
docker-compose exec app bash
```

### Development with Docker
```bash
# Start with file watching (development)
docker-compose up --build

# Start only specific services
docker-compose up -d mysql redis chromadb

# View service status
docker-compose ps

# Clean up everything (including volumes)
docker-compose down -v
```

---

## 🏗️ Project Structure

```
AI-Job-Application-Coach/
├── app/
│   ├── agents/          # Individual AI agents
│   ├── tools/           # Utility functions and tools
│   ├── graph/           # LangGraph orchestration
│   ├── rag/             # Knowledge retrieval system
│   └── main.py          # FastAPI application
├── tests/               # Unit and integration tests
├── evaluation/          # Evaluation metrics and benchmarks
├── scripts/             # Database and utility scripts
├── docs/               # Documentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 🤖 Features

### Core Capabilities

- **Resume Analysis**: AI-powered resume review with actionable feedback
- **Interview Practice**: Mock interviews with role-specific questions and evaluation
- **Job Discovery**: Search for relevant job opportunities (mock implementation)
- **Career Advice**: RAG-powered guidance on career topics
- **Application Tracking**: Manage your job application pipeline
- **Persistent Memory**: Learn your preferences and history across sessions

### Multi-Agent Architecture

1. **Router Agent**: Classifies user intent and routes to appropriate specialist
2. **Resume Agent**: Analyzes resumes and provides improvement suggestions
3. **Interview Agent**: Conducts practice interviews and evaluates responses
4. **Job Search Agent**: Finds relevant job opportunities
5. **Knowledge Agent**: Retrieves career advice using RAG
6. **Memory Agent**: Maintains user profiles and conversation history

---

## 📚 API Endpoints

### Resume Analysis
- `POST /resume` — Analyse resume and get structured feedback
- `POST /resume/improve` — Concrete improvement suggestions with rewritten bullets
- `POST /resume/audit` — Dispatch async multi-step audit (returns 202 + task ID)

### Interview Practice
- `POST /interview/start` — Start mock-interview session
- `POST /interview/answer` — Submit answer and get STAR evaluation
- `GET  /interview/questions/{title}` — Generate questions without a session
- `POST /interview/report` — Dispatch async performance report

### Job Search
- `POST /jobs/search` — Search with geocoding + LLM results
- `GET  /jobs/location/{city}` — Geocode city + nearby companies
- `POST /jobs/match` — Profile-scored job matching

### Career Advice  
- `POST /ask` — RAG-powered career Q&A

### Application Tracking
- `GET    /applications` — List applications (optional `?status=` filter)
- `POST   /applications` — Create new application
- `PUT    /applications/{id}` — Update with status-workflow validation
- `DELETE /applications/{id}` — Delete application
- `GET    /applications/follow-ups` — Overdue follow-up reminders
- `POST   /applications/batch-update` — Dispatch async stale/overdue check

### Unified Chat
- `POST /chat` — Natural-language endpoint routed through LangGraph

### Async Tasks
- `GET /tasks/{task_id}/status` — Poll Celery task progress/result
- `GET /result/{task_id}` — Legacy alias

### User & Memory
- `GET  /user/{id}/profile` — User profile data
- `GET  /user/{id}/context` — Conversation context for agents
- `GET  /user/{id}/insights` — Analytical conversation patterns

### System
- `GET /health` — Subsystem health (database + Redis)
- `GET /docs` — Interactive Swagger UI

---

## 🧪 Testing the System

### 1. Resume Review
```bash
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "John Smith\nSoftware Engineer with 5 years experience...",
    "job_description": "Senior Backend Developer position..."
  }'
```

### 2. Start Interview Practice
```bash
curl -X POST http://localhost:8000/interview/start \
  -H "Content-Type: application/json" \
  -d '{
    "role": "Software Engineer",
    "level": "senior"
  }'
```

### 3. Career Advice
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How should I prepare for salary negotiation?"
  }'
```

---

## 🔧 Development Status

### ✅ Phase 1: Foundation (Completed)
- [x] Project structure and dependencies
- [x] Database schema and utilities  
- [x] Basic FastAPI application with all endpoints
- [x] LangGraph workflow foundation
- [x] Environment configuration

### ✅ Phase 2: Agents (Completed)
- [x] Resume analysis agent with LLM integration
- [x] Interview agent with question generation & STAR evaluation
- [x] Knowledge agent with RAG implementation
- [x] Memory agent with DB persistence & LLM summarization
- [x] Job search agent with geocoding & LLM results

### ✅ Phase 3: Orchestration (Completed)
- [x] Router agent with LLM intent classification
- [x] LangGraph StateGraph workflow (6 nodes)
- [x] Cross-agent communication & state sharing
- [x] Unified `/chat` endpoint  
- [x] 83 tests passing

### ✅ Phase 4: Deployment & Async (Completed)
- [x] Docker containerisation (MySQL, Redis, ChromaDB, app, Celery worker)
- [x] Production config via Pydantic BaseSettings
- [x] Structured logging with per-request correlation IDs
- [x] Security middleware (API key auth, rate limiting, input validation)
- [x] Complete application tracker CRUD with status workflow
- [x] Celery infrastructure with Redis broker
- [x] Async resume audit (multi-step Celery task with progress)
- [x] Background interview reports & batch application checks
- [x] Global exception handler & graceful degradation
- [x] Enhanced health check with subsystem reporting

### 📋 Phase 5: Evaluation & Polish (Upcoming)
- [ ] Evaluation framework and metrics
- [ ] Prompt optimisation & A/B testing
- [ ] Comprehensive error handling refinement
- [ ] Final documentation and demo preparation

---

## 🛠️ Technology Stack

- **Framework**: LangChain 0.2, LangGraph
- **API**: FastAPI 0.104, Uvicorn
- **Database**: MySQL 8.0 
- **Vector DB**: ChromaDB 0.4
- **LLM**: OpenAI GPT-4o-mini
- **Cache/Queue**: Redis 7 + Celery 5.3
- **Security**: API-key auth, rate limiting, input validation
- **Containers**: Docker, Docker Compose
- **Testing**: pytest (83+ tests)

### Celery Workers (Async Tasks)

To enable async features (resume audit, interview reports):

```bash
# Start a Celery worker (requires Redis running)
celery -A app.celery_worker worker --loglevel=info

# Or via Docker Compose (starts automatically)
docker compose up celery_worker
```

### Security Configuration

```bash
# In .env — set API_KEY to enable authentication
API_KEY=your-secret-key

# Rate limiting (requests per minute per IP)
RATE_LIMIT_PER_MINUTE=60
```

When `API_KEY` is unset (default), authentication is bypassed for development.
Public paths (`/health`, `/docs`, `/openapi.json`, `/redoc`) are always accessible.

---

## 📖 Documentation

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Detailed technical implementation
- [Implementation Checklist](docs/IMPLEMENTATION_CHECKLIST.md) - Development progress tracking
- [Project Plan](docs/PLAN.md) - Complete project overview and architecture

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Troubleshooting

### Docker Issues

**Docker not starting:**
- Ensure Docker Desktop is installed and running
- On Windows, enable WSL 2 backend
- Check Docker has sufficient memory allocated (4GB+ recommended)

**Services not connecting:**
```bash
# Check service health
docker-compose ps

# View service logs
docker-compose logs mysql
docker-compose logs app

# Restart problematic service
docker-compose restart mysql
```

**Port conflicts:**
```bash
# If ports 8000, 3306, 6379, or 8001 are busy
# Stop conflicting services or modify ports in docker-compose.yml
netstat -tulpn | grep :8000  # Linux
netstat -an | findstr :8000  # Windows
```

### Manual Setup Issues

**Database Connection Error**
- Ensure MySQL server is running
- Verify credentials in `.env` file
- Check if `job_coach` database exists
- Test connection: `mysql -u jobcoach -p -h localhost job_coach`

**OpenAI API Error** 
- Verify API key is set correctly in `.env`
- Check API key has sufficient credits
- Ensure network connectivity
- Test: `curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models`

**Import Errors**
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version is 3.9+: `python --version`

**ChromaDB Issues**
- Ensure ChromaDB directory has write permissions
- Clear ChromaDB data: `rm -rf chroma/` (will lose vector data)
- For Docker: `docker-compose down -v` to reset volumes

### Performance Issues

**Slow API responses:**
- Check database connection pool settings
- Monitor Docker container resources: `docker stats`
- Increase Docker memory allocation
- Use `docker-compose logs -f app` to check for bottlenecks

**Memory usage:**
```bash
# Monitor container memory
docker stats --no-stream

# Check system resources
free -h    # Linux
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:table  # Windows
```

### Getting Help

- Check the [documentation](docs/) for detailed implementation guides
- Open an issue if you encounter bugs
- Review the API docs at `http://localhost:8000/docs` when running

---

**🎯 Current Status**: Phases 1–4 complete — ready for Phase 5 evaluation & polish!