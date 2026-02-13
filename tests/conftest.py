"""
Shared pytest fixtures for Phase 2 agent tests.

Provides sample data, mock LLM helpers, and reusable test constants.
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables before any agent imports
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ------------------------------------------------------------------ #
#  Sample data fixtures
# ------------------------------------------------------------------ #

SAMPLE_RESUME = """
Jane Smith
Senior Backend Engineer | jane.smith@email.com | Tel Aviv, Israel

SUMMARY
Experienced backend engineer with 7 years building scalable distributed systems.
Proficient in Python, Go, and cloud-native architectures.

EXPERIENCE
Senior Backend Engineer — CyberTech Ltd. (2021–Present)
- Architected event-driven microservices platform processing 1M events/day using Kafka
- Led migration from AWS to hybrid cloud, reducing infrastructure costs by 35%
- Mentored team of 4 junior developers, established code review standards

Backend Developer — DataFlow Startup (2018–2021)
- Built REST APIs in Python/FastAPI handling 50K concurrent users
- Implemented CI/CD with GitHub Actions, cutting deployment time by 70%
- Designed PostgreSQL sharding strategy for multi-region SaaS product

Junior Developer — WebSolutions (2016–2018)
- Developed internal tools using Python and Django
- Created automated testing framework achieving 90% code coverage

SKILLS
Python, Go, FastAPI, Django, AWS, GCP, Docker, Kubernetes, PostgreSQL, Redis, Kafka, gRPC

EDUCATION
B.Sc. Computer Science — Technion (2016)
"""

SAMPLE_JOB_DESCRIPTION = """
Senior Backend Engineer — FinTech Startup (Tel Aviv)

We're looking for a senior backend engineer to build our real-time payment
processing platform.

Requirements:
- 5+ years backend development experience
- Strong Python and distributed systems knowledge
- Experience with event-driven architecture (Kafka, RabbitMQ)
- SQL databases (PostgreSQL preferred)
- Cloud infrastructure (AWS/GCP)
- CI/CD pipeline experience

Nice to have:
- Go or Rust experience
- Financial domain knowledge
- Kubernetes / container orchestration
- gRPC / Protocol Buffers
"""

MINIMAL_RESUME = "John Doe\nSoftware Developer\njohn@email.com"

EMPTY_RESUME = ""


@pytest.fixture
def sample_resume():
    return SAMPLE_RESUME


@pytest.fixture
def sample_job_description():
    return SAMPLE_JOB_DESCRIPTION


@pytest.fixture
def minimal_resume():
    return MINIMAL_RESUME


@pytest.fixture
def sample_question():
    """A typical interview question dict."""
    return {
        "id": "q1",
        "question": "Tell me about a time you had to debug a complex production issue.",
        "type": "behavioral",
        "difficulty": "medium",
        "key_points": [
            "Systematic debugging approach",
            "Root cause analysis",
            "Team collaboration",
            "Lessons learned",
        ],
    }


@pytest.fixture
def sample_answer():
    """A reasonable candidate answer for interview evaluation."""
    return (
        "At CyberTech, our payment processing service started failing intermittently "
        "during peak hours. I led the investigation by first checking our monitoring "
        "dashboards in Grafana, which showed memory spikes correlating with failures. "
        "I narrowed it down to a connection pool leak in our PostgreSQL adapter — "
        "connections weren't being released after timeouts. I implemented connection "
        "lifecycle management with proper context managers, added circuit breakers, "
        "and set up alerts. The fix reduced error rates from 2% to 0.01%. "
        "I then documented the incident and created a runbook for the team."
    )


@pytest.fixture
def weak_answer():
    """A weak/vague candidate answer for edge-case testing."""
    return "I fixed a bug once. It was hard but I figured it out."


# ------------------------------------------------------------------ #
#  Mock LLM helper
# ------------------------------------------------------------------ #

def make_mock_llm_response(content: str):
    """Create a mock LLM response object with a .content attribute."""
    mock = MagicMock()
    mock.content = content
    return mock
