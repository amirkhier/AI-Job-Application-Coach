# Database Documentation

## Overview

The AI Job Application Coach uses **MySQL 8.0** as its primary relational database. The database stores user profiles, conversation history, job application records, and interview practice sessions. All data is managed through the `DatabaseManager` class located in `app/tools/database.py`.

## Infrastructure

| Property          | Value                        |
|-------------------|------------------------------|
| Engine            | MySQL 8.0                    |
| Database name     | `job_coach`                  |
| Container         | `job_coach_mysql`            |
| Internal port     | 3306                         |
| External port     | 3307 (host)                  |
| Charset           | utf8mb4                      |
| Docker volume     | `mysql_data`                 |

### Environment Variables

| Variable          | Default (Docker)   | Description                |
|-------------------|--------------------|----------------------------|
| `MYSQL_HOST`      | `mysql`            | Hostname inside Docker     |
| `MYSQL_PORT`      | `3306`             | Internal container port    |
| `MYSQL_USER`      | `jobcoach`         | Application DB user        |
| `MYSQL_PASSWORD`  | `jobcoach123`      | Application DB password    |
| `MYSQL_DATABASE`  | `job_coach`        | Database name              |

---

## Schema

The schema is defined in `scripts/setup_db.sql` and contains four tables.

### Entity-Relationship Diagram

```
┌────────────┐       ┌──────────────────┐
│   users    │──1:N──│  conversations   │
│            │       └──────────────────┘
│            │       ┌──────────────────┐
│            │──1:N──│  applications    │
│            │       └──────────────────┘
│            │       ┌──────────────────────┐
│            │──1:N──│  interview_sessions  │
└────────────┘       └──────────────────────┘
```

All child tables reference `users(id)` with `ON DELETE CASCADE`.

---

### `users`

Stores user accounts, career profiles, and preferences.

| Column         | Type           | Constraints              | Description                                     |
|----------------|----------------|--------------------------|-------------------------------------------------|
| `id`           | INT            | PK, AUTO_INCREMENT       | Unique user identifier                          |
| `email`        | VARCHAR(255)   | UNIQUE                   | User email address                              |
| `created_at`   | TIMESTAMP      | DEFAULT CURRENT_TIMESTAMP| Account creation time                           |
| `updated_at`   | TIMESTAMP      | AUTO-UPDATE              | Last modification time                          |
| `profile_data` | JSON           | NULLABLE                 | Career profile (name, skills, experience, etc.) |
| `preferences`  | JSON           | NULLABLE                 | User preferences (roles, location, etc.)        |

**Indexes:** `idx_email`, `idx_created_at`

#### `profile_data` JSON Structure

```json
{
  "name": "Test User",
  "skills": ["Python", "FastAPI", "React"],
  "experience_years": 2,
  "experience_level": "junior",
  "career_goals": "Become a senior backend developer",
  "target_roles": ["Backend Developer", "Data Engineer"]
}
```

#### `preferences` JSON Structure

```json
{
  "preferred_roles": ["Software Engineer", "Backend Developer"],
  "location": "Remote"
}
```

---

### `conversations`

Stores every user–agent interaction for memory persistence and analytics.

| Column        | Type          | Constraints              | Description                          |
|---------------|---------------|--------------------------|--------------------------------------|
| `id`          | INT           | PK, AUTO_INCREMENT       | Unique conversation turn ID          |
| `user_id`     | INT           | FK → users(id), NOT NULL | Owner of the conversation            |
| `session_id`  | VARCHAR(255)  | NOT NULL                 | Groups turns into a session          |
| `message`     | TEXT          | NOT NULL                 | User's message                       |
| `response`    | TEXT          | NULLABLE                 | Agent's response                     |
| `intent`      | VARCHAR(50)   | NULLABLE                 | Detected intent (resume, interview…) |
| `agent_used`  | VARCHAR(50)   | NULLABLE                 | Which agent handled the request      |
| `created_at`  | TIMESTAMP     | DEFAULT CURRENT_TIMESTAMP| When the turn occurred               |
| `metadata`    | JSON          | NULLABLE                 | Extra data (summary, context, etc.)  |

**Indexes:** `idx_user_session(user_id, session_id)`, `idx_created_at`, `idx_intent`, `idx_agent`

#### `metadata` JSON Structure (example)

```json
{
  "summary": "User asked about resume formatting",
  "tokens_used": 450,
  "confidence": 0.92
}
```

---

### `applications`

Tracks job applications through the hiring pipeline.

| Column             | Type                                                        | Constraints              | Description                    |
|--------------------|-------------------------------------------------------------|--------------------------|--------------------------------|
| `id`               | INT                                                         | PK, AUTO_INCREMENT       | Unique application ID          |
| `user_id`          | INT                                                         | FK → users(id), NOT NULL | Applicant                      |
| `company_name`     | VARCHAR(255)                                                | NOT NULL                 | Company name                   |
| `position_title`   | VARCHAR(255)                                                | NOT NULL                 | Job title                      |
| `job_url`          | TEXT                                                        | NULLABLE                 | Link to the job posting        |
| `status`           | ENUM('applied','interviewing','offer','rejected','withdrawn')| DEFAULT 'applied'       | Current pipeline stage         |
| `application_date` | DATE                                                        | NOT NULL                 | When the application was sent  |
| `follow_up_date`   | DATE                                                        | NULLABLE                 | Scheduled follow-up date       |
| `notes`            | TEXT                                                        | NULLABLE                 | Free-form notes                |
| `created_at`       | TIMESTAMP                                                   | DEFAULT CURRENT_TIMESTAMP| Record creation time           |
| `updated_at`       | TIMESTAMP                                                   | AUTO-UPDATE              | Last modification time         |

**Indexes:** `idx_user_id`, `idx_status`, `idx_application_date`, `idx_company_name`

---

### `interview_sessions`

Stores mock interview practice sessions and performance data.

| Column         | Type          | Constraints              | Description                              |
|----------------|---------------|--------------------------|------------------------------------------|
| `id`           | INT           | PK, AUTO_INCREMENT       | Unique session record ID                 |
| `user_id`      | INT           | FK → users(id), NOT NULL | Interviewee                              |
| `session_id`   | VARCHAR(255)  | UNIQUE, NOT NULL         | UUID identifying the session             |
| `role`         | VARCHAR(255)  | NOT NULL                 | Target role for the interview            |
| `level`        | VARCHAR(50)   | DEFAULT 'mid'            | Difficulty level                         |
| `questions`    | JSON          | —                        | Array of generated questions             |
| `answers`      | JSON          | —                        | Array of user's answers                  |
| `feedback`     | JSON          | —                        | Structured feedback per answer           |
| `score`        | DECIMAL(3,2)  | NULLABLE                 | Overall performance score (0.00 – 9.99)  |
| `completed_at` | TIMESTAMP     | NULLABLE                 | When the session was completed           |
| `created_at`   | TIMESTAMP     | DEFAULT CURRENT_TIMESTAMP| Session creation time                    |

**Indexes:** `idx_user_id`, `idx_session_id`, `idx_role`, `idx_score`, `idx_created_at`

---

## DatabaseManager API

The `DatabaseManager` class (`app/tools/database.py`) provides all database operations. A singleton instance is created at module level and accessed via `get_db()`.

### Connection Management

| Method              | Description                                        |
|---------------------|----------------------------------------------------|
| `connect()`         | Establish MySQL connection (auto-reconnect)        |
| `disconnect()`      | Close the active connection                        |
| `ensure_connection()` | Reconnect if the connection was lost             |

### Low-Level Execution

| Method                                | Returns          | Description                                           |
|---------------------------------------|------------------|-------------------------------------------------------|
| `execute_query(query, params)`        | `List[Dict]`     | Run a SELECT, return rows as dictionaries             |
| `execute_update(query, params)`       | `int`            | Run INSERT/UPDATE/DELETE, return lastrowid or rowcount|
| `execute_many(query, params_list)`    | `int`            | Batch execute with multiple parameter sets            |

### User Operations

| Method                                          | Returns      | Description                              |
|-------------------------------------------------|--------------|------------------------------------------|
| `create_user(email, profile_data, preferences)` | `int`        | Create user, return ID                   |
| `get_user(user_id)`                             | `Dict`       | Fetch user with parsed JSON fields       |
| `update_user_profile(user_id, profile_data)`    | `bool`       | Update `profile_data` JSON column        |

### Conversation Operations

| Method                                                              | Returns      | Description                                    |
|---------------------------------------------------------------------|--------------|------------------------------------------------|
| `save_conversation(user_id, session_id, message, response, ...)`    | `int`        | Insert a conversation turn, return ID          |
| `get_conversation_history(user_id, session_id=None, limit=20)`      | `List[Dict]` | Fetch recent conversations (newest first)      |

### Application Operations

| Method                                                              | Returns      | Description                                    |
|---------------------------------------------------------------------|--------------|------------------------------------------------|
| `create_application(user_id, company_name, position_title, ...)`    | `int`        | Create application, return ID                  |
| `get_applications(user_id, status=None)`                            | `List[Dict]` | List applications, optionally filter by status |
| `get_application_by_id(application_id)`                             | `Dict`       | Fetch single application by ID                 |
| `update_application_status(application_id, status, notes=None)`     | `bool`       | Update status (and optionally notes)           |
| `update_application(application_id, **fields)`                      | `bool`       | Update arbitrary allowed fields                |
| `delete_application(application_id)`                                | `bool`       | Hard-delete an application record              |

### Interview Session Operations

| Method                                                                      | Returns      | Description                                    |
|-----------------------------------------------------------------------------|--------------|------------------------------------------------|
| `create_interview_session(user_id, session_id, role, level)`                | `int`        | Create session with empty Q&A arrays           |
| `update_interview_session(session_id, questions, answers, feedback, score)` | `bool`       | Update session data and optionally complete it |
| `get_interview_session(session_id)`                                         | `Dict`       | Fetch session with parsed JSON fields          |

---

## Dependency Injection

The FastAPI app uses a `Depends(get_database)` pattern to inject `DatabaseManager` into route handlers:

```python
from app.tools.database import DatabaseManager

def get_database() -> DatabaseManager:
    db = DatabaseManager()
    db.connect()
    return db

@app.post("/applications")
async def create_application(request: ..., db: DatabaseManager = Depends(get_database)):
    ...
```

---

## Transactions

For operations requiring explicit transaction control, use the `DatabaseTransaction` context manager:

```python
from app.tools.database import DatabaseTransaction, db

with DatabaseTransaction(db) as tx:
    tx.execute_update("INSERT INTO ...", params)
    tx.execute_update("UPDATE ...", params)
    # auto-rollback on exception, otherwise committed by execute_update
```

Note: `execute_update` calls `connection.commit()` after each statement. For multi-statement atomic operations, manual transaction handling would need to be added.

---

## Seed Data

The schema script inserts one default user for development/testing:

```sql
INSERT IGNORE INTO users (id, email, profile_data, preferences) VALUES
(1, 'test@example.com',
 JSON_OBJECT('name', 'Test User', 'skills', JSON_ARRAY('Python', 'FastAPI'), 'experience_years', 5),
 JSON_OBJECT('preferred_roles', JSON_ARRAY('Software Engineer', 'Backend Developer'), 'location', 'Remote')
);
```

---

## Docker Setup

The MySQL service is defined in `docker-compose.yml`:

```yaml
mysql:
  image: mysql:8.0
  container_name: job_coach_mysql
  environment:
    MYSQL_ROOT_PASSWORD: rootpassword
    MYSQL_DATABASE: job_coach
    MYSQL_USER: jobcoach
    MYSQL_PASSWORD: jobcoach123
  ports:
    - "3307:3306"
  volumes:
    - mysql_data:/var/lib/mysql
    - ./scripts/setup_db.sql:/docker-entrypoint-initdb.d/setup.sql
  healthcheck:
    test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
    interval: 10s
    timeout: 5s
    retries: 5
```

The schema is automatically applied on first container creation via the `docker-entrypoint-initdb.d` mount.

### Connecting from Host

```bash
mysql -h 127.0.0.1 -P 3307 -u jobcoach -pjobcoach123 job_coach
```

---

## Health Check

The `/health` endpoint verifies MySQL connectivity:

```json
{
  "status": "healthy",
  "checks": {
    "database": true,
    "redis": true,
    "chromadb": true
  }
}
```

The database check attempts to fetch user ID 1. If the query succeeds, `database: true`.
