-- MySQL schema for AI Job Application Coach
-- Run this script to set up the database structure

CREATE DATABASE IF NOT EXISTS job_coach;
USE job_coach;

-- Users table for profile management
CREATE TABLE IF NOT EXISTS users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    profile_data JSON,
    preferences JSON,
    INDEX idx_email (email),
    INDEX idx_created_at (created_at)
);

-- Conversations table for memory persistence
CREATE TABLE IF NOT EXISTS conversations (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    response TEXT,
    intent VARCHAR(50),
    agent_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_session (user_id, session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_intent (intent),
    INDEX idx_agent (agent_used)
);

-- Applications table for job tracking
CREATE TABLE IF NOT EXISTS applications (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    position_title VARCHAR(255) NOT NULL,
    job_url TEXT,
    status ENUM('applied', 'interviewing', 'offer', 'rejected', 'withdrawn') DEFAULT 'applied',
    application_date DATE NOT NULL,
    follow_up_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_application_date (application_date),
    INDEX idx_company_name (company_name)
);

-- Interview sessions for practice tracking
CREATE TABLE IF NOT EXISTS interview_sessions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    session_id VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(255) NOT NULL,
    level VARCHAR(50) DEFAULT 'mid',
    questions JSON,
    answers JSON,
    feedback JSON,
    score DECIMAL(3,2),
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_role (role),
    INDEX idx_score (score),
    INDEX idx_created_at (created_at)
);

-- Create a default user for testing
INSERT IGNORE INTO users (id, email, profile_data, preferences) VALUES 
(1, 'test@example.com', 
 JSON_OBJECT('name', 'Test User', 'skills', JSON_ARRAY('Python', 'FastAPI'), 'experience_years', 5),
 JSON_OBJECT('preferred_roles', JSON_ARRAY('Software Engineer', 'Backend Developer'), 'location', 'Remote')
);

-- Show tables to confirm creation
SHOW TABLES;