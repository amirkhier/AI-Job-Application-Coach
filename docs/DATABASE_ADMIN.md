# Database Administration Guide

## Browser-Based Management (Adminer)

The project includes [Adminer](https://www.adminer.org/) — a lightweight, browser-based database management UI running as a Docker container.

### Starting Adminer

```bash
docker-compose up -d adminer
```

### Accessing Adminer

Open **http://localhost:8080** in your browser and log in with:

| Field    | Value         |
|----------|---------------|
| System   | MySQL         |
| Server   | `mysql`       |
| Username | `jobcoach`    |
| Password | `jobcoach123` |
| Database | `job_coach`   |

### Features

- Browse and edit table data
- Run SQL queries
- Export/import data (SQL, CSV, JSON)
- View table structure and indexes
- Manage users and permissions

---

## Command-Line Access

### Via Docker (recommended)

Run queries directly against the MySQL container:

```bash
docker exec job_coach_mysql mysql -ujobcoach -pjobcoach123 job_coach -e "YOUR SQL HERE"
```

Interactive shell:

```bash
docker exec -it job_coach_mysql mysql -ujobcoach -pjobcoach123 job_coach
```

### From Host Machine

Connect using the exposed host port (3307):

```bash
mysql -h 127.0.0.1 -P 3307 -u jobcoach -pjobcoach123 job_coach
```

---

## Common Queries

### View All Tables

```sql
SHOW TABLES;
```

### User Profile

```sql
-- View user profile data
SELECT id, email, JSON_PRETTY(profile_data) AS profile, JSON_PRETTY(preferences) AS prefs
FROM users WHERE id = 1\G

-- Update experience years
UPDATE users
SET profile_data = JSON_SET(profile_data, '$.experience_years', 2)
WHERE id = 1;

-- Update experience level
UPDATE users
SET profile_data = JSON_SET(profile_data, '$.experience_level', 'junior')
WHERE id = 1;

-- Add a skill
UPDATE users
SET profile_data = JSON_ARRAY_APPEND(profile_data, '$.skills', 'Docker')
WHERE id = 1;
```

### Conversations

```sql
-- Recent conversations
SELECT id, intent, agent_used, LEFT(message, 80) AS message, created_at
FROM conversations
ORDER BY created_at DESC LIMIT 10;

-- Count by agent
SELECT agent_used, COUNT(*) AS total
FROM conversations
GROUP BY agent_used
ORDER BY total DESC;

-- Count by intent
SELECT intent, COUNT(*) AS total
FROM conversations
GROUP BY intent
ORDER BY total DESC;
```

### Applications

```sql
-- All applications
SELECT id, company_name, position_title, status, application_date
FROM applications
ORDER BY application_date DESC;

-- Applications by status
SELECT status, COUNT(*) AS total
FROM applications
GROUP BY status;

-- Update application status
UPDATE applications SET status = 'interviewing' WHERE id = 1;
```

### Interview Sessions

```sql
-- All sessions
SELECT id, session_id, role, level, score, completed_at, created_at
FROM interview_sessions
ORDER BY created_at DESC;

-- Average score by role
SELECT role, AVG(score) AS avg_score, COUNT(*) AS sessions
FROM interview_sessions
WHERE score IS NOT NULL
GROUP BY role;
```

### Data Counts (Quick Health Check)

```sql
SELECT 'users' AS tbl, COUNT(*) AS cnt FROM users
UNION ALL SELECT 'conversations', COUNT(*) FROM conversations
UNION ALL SELECT 'applications', COUNT(*) FROM applications
UNION ALL SELECT 'interview_sessions', COUNT(*) FROM interview_sessions;
```

---

## Connection Details

| Property       | Docker Internal | Host Access        |
|----------------|-----------------|--------------------|
| Host           | `mysql`         | `localhost`        |
| Port           | `3306`          | `3307`             |
| Username       | `jobcoach`      | `jobcoach`         |
| Password       | `jobcoach123`   | `jobcoach123`      |
| Database       | `job_coach`     | `job_coach`        |
| Root Password  | —               | `rootpassword`     |
| Container      | —               | `job_coach_mysql`  |
| Adminer UI     | —               | `http://localhost:8080` |

---

## Backup & Restore

### Export (Backup)

```bash
docker exec job_coach_mysql mysqldump -ujobcoach -pjobcoach123 job_coach > backup.sql
```

### Import (Restore)

```bash
docker exec -i job_coach_mysql mysql -ujobcoach -pjobcoach123 job_coach < backup.sql
```

### Export Specific Table

```bash
docker exec job_coach_mysql mysqldump -ujobcoach -pjobcoach123 job_coach applications > applications.sql
```

---

## Reset Database

To wipe all data and re-initialize from the schema:

```bash
# Stop services
docker-compose down

# Remove the MySQL volume
docker volume rm ai-job-application-coach_mysql_data

# Restart — schema will be re-applied from setup_db.sql
docker-compose up -d mysql
```
