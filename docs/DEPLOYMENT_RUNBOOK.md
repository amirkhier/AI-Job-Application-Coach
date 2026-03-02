# Deployment Runbook

## Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ (3.12 recommended) | Application runtime |
| Docker + Docker Compose | Latest | Container orchestration |
| MySQL | 8.0 | Application database |
| Redis | 7+ | Celery broker / result backend |
| ChromaDB | Latest | Vector database for RAG |
| OpenAI API Key | — | GPT-4o-mini access |

---

## 1. Docker Deployment (Recommended)

### 1.1 Initial Setup

```bash
# Clone the repository
git clone <repo-url>
cd AI-Job-Application-Coach

# Create environment file
cp .env.docker .env

# Set your OpenAI API key
# Edit .env → OPENAI_API_KEY=sk-...
```

### 1.2 Start All Services

```bash
# Linux / macOS
./docker-setup.sh

# Windows
docker-setup.bat

# Or manually:
docker-compose up -d --build
```

This starts 4 containers:

| Container | Image | Port (host) | Purpose |
|-----------|-------|-------------|---------|
| `job_coach_app` | Custom (Dockerfile) | 8000 | FastAPI application |
| `job_coach_mysql` | mysql:8.0 | 3307 | MySQL database |
| `job_coach_redis` | redis:7-alpine | 6380 | Celery broker |
| `job_coach_chromadb` | chromadb/chroma | 8001 | Vector database |

### 1.3 Verify Health

```bash
curl http://localhost:8000/health

# Expected:
# {"status":"healthy","service":"AI Job Application Coach",...}
```

### 1.4 Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| ChromaDB | http://localhost:8001 |
| MySQL | `mysql -h 127.0.0.1 -P 3307 -u jobcoach -pjobcoach123 job_coach` |

### 1.5 Management Commands

```bash
# View logs
docker-compose logs -f app

# Restart application only
docker-compose restart app

# Stop all services
docker-compose down

# Stop and remove volumes (full reset)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build app
```

---

## 2. Manual Deployment

### 2.1 Python Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR: venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2.2 Database Setup

```bash
# Create database and tables
mysql -u root -p < scripts/setup_db.sql
```

### 2.3 Environment Variables

Create `.env` in the project root:

```env
# Required
OPENAI_API_KEY=sk-...

# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=job_coach

# Redis (for async tasks)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Application
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
LOG_FORMAT=text
API_PORT=8000

# Optional: API key authentication
# API_KEY=your-secret-api-key

# Optional: Rate limiting
RATE_LIMIT_PER_MINUTE=60
```

### 2.4 Start the Application

```bash
# Development (with auto-reload)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2.5 Start Celery Worker (for async tasks)

```bash
celery -A app.celery_worker worker --loglevel=info
```

### 2.6 Initialize RAG Database

```bash
python -m app.rag.create_database
```

---

## 3. Production Configuration

### 3.1 Environment Settings

```env
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=WARNING
LOG_FORMAT=json
API_KEY=<strong-random-key>
ALLOWED_ORIGINS=https://your-domain.com
RATE_LIMIT_PER_MINUTE=30
```

### 3.2 Security Checklist

- [ ] Set `API_KEY` to a strong random value
- [ ] Set `ALLOWED_ORIGINS` to specific domains (not `*`)
- [ ] Use strong MySQL passwords
- [ ] Run behind a reverse proxy (nginx/Caddy) with TLS
- [ ] Set `DEBUG=False`
- [ ] Use `LOG_FORMAT=json` for structured log aggregation
- [ ] Rotate `OPENAI_API_KEY` periodically
- [ ] Restrict database access to application user only

### 3.3 Reverse Proxy (nginx example)

```nginx
server {
    listen 443 ssl;
    server_name api.yourcoach.com;

    ssl_certificate     /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

---

## 4. Monitoring

### 4.1 Health Endpoint

Poll `GET /health` every 30 seconds. Alert when:
- `status` is `degraded`
- `database_connected` is `false`

### 4.2 Logging

| Format | Use Case | Example |
|--------|----------|---------|
| `text` | Development | `2026-03-02 20:00:00 [INFO] ResumeAgent initialised` |
| `json` | Production | `{"timestamp":"...","level":"INFO","message":"...","request_id":"..."}` |

All requests include `X-Request-ID` for correlation.

### 4.3 Key Metrics to Monitor

| Metric | Source | Threshold |
|--------|--------|-----------|
| Response latency (p95) | Application logs | < 5s single agent, < 10s graph |
| Error rate | HTTP 5xx responses | < 1% |
| Rate limit rejections | 429 responses | Investigate if > 10% |
| OpenAI API errors | Application logs | Any occurrence |
| Database connection failures | Health check | Any occurrence |
| Memory usage | Container stats | < 1GB per worker |

---

## 5. Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OPENAI_API_KEY not set` | Missing env var | Add key to `.env` |
| `Can't connect to MySQL` | Database not running | Start MySQL: `docker-compose up -d mysql` |
| Health status `degraded` | Optional dependency down | Check Redis/ChromaDB; app still works |
| `429 Too Many Requests` | Rate limit hit | Wait 1 minute or increase `RATE_LIMIT_PER_MINUTE` |
| Celery tasks stuck `PENDING` | Redis not running | Start Redis: `docker-compose up -d redis` |
| ChromaDB collection empty | RAG not initialized | Run `python -m app.rag.create_database` |
| Import errors | Missing dependencies | `pip install -r requirements.txt` |

### Database Reset

```bash
# Docker
docker-compose down -v
docker-compose up -d

# Manual
mysql -u root -p -e "DROP DATABASE job_coach; SOURCE scripts/setup_db.sql;"
```

### Log Inspection

```bash
# Docker
docker-compose logs -f --tail=100 app

# Manual
# Logs go to stdout; redirect with:
python -m uvicorn app.main:app 2>&1 | tee app.log
```

---

## 6. Backup & Recovery

### Database Backup

```bash
# Export
docker exec job_coach_mysql mysqldump -u root -prootpassword job_coach > backup.sql

# Import
docker exec -i job_coach_mysql mysql -u root -prootpassword job_coach < backup.sql
```

### ChromaDB Data

ChromaDB data is persisted in the `chromadb_data` Docker volume. Back up the volume:

```bash
docker run --rm -v chromadb_data:/data -v $(pwd):/backup alpine tar czf /backup/chromadb_backup.tar.gz /data
```
