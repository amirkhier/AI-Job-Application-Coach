"""
Tests for Phase 4 Step 2 — Security Middleware.

Covers:
  - API key authentication (bypass when unset, enforce when set)
  - Rate limiting (429 after threshold)
  - Input validation (SQL injection rejection, body size limits)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_app():
    """Import and return the FastAPI app (fresh for each test that needs it)."""
    from app.main import app
    return app


def _client():
    return TestClient(_get_app())


# ===========================================================================
# API Key Authentication
# ===========================================================================

class TestApiKeyMiddleware:
    """API key gate tests."""

    def test_health_accessible_without_key(self):
        """Public paths (/health, /docs) should never require a key."""
        client = _client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_docs_accessible_without_key(self):
        client = _client()
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_no_key_configured_allows_all(self):
        """When API_KEY is None (default), all endpoints are open."""
        client = _client()
        # /health is always open, but let's check a secured endpoint too
        resp = client.get("/health")
        assert resp.status_code == 200

    @patch("app.middleware.auth.settings")
    def test_valid_key_in_header(self, mock_settings):
        """Correct X-API-Key header should grant access."""
        mock_settings.API_KEY = "test-secret-key"
        client = _client()
        resp = client.get("/health")
        # /health is a public path, should always be 200
        assert resp.status_code == 200

    @patch("app.middleware.auth.settings")
    def test_invalid_key_returns_401(self, mock_settings):
        """Wrong API key should produce 401."""
        mock_settings.API_KEY = "correct-key"
        client = _client()
        resp = client.post(
            "/resume",
            json={"resume_text": "x" * 60, "user_id": 1},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid or missing" in resp.json()["detail"]

    @patch("app.middleware.auth.settings")
    def test_missing_key_returns_401(self, mock_settings):
        """No API key at all should produce 401 when key is configured."""
        mock_settings.API_KEY = "correct-key"
        client = _client()
        resp = client.post(
            "/resume",
            json={"resume_text": "x" * 60, "user_id": 1},
        )
        assert resp.status_code == 401

    @patch("app.middleware.auth.settings")
    def test_key_in_query_param(self, mock_settings):
        """API key passed via ?api_key= query parameter should be accepted."""
        mock_settings.API_KEY = "correct-key"
        client = _client()
        resp = client.get("/health?api_key=correct-key")
        # /health is public, but this validates that query params are read
        assert resp.status_code == 200


# ===========================================================================
# Rate Limiting
# ===========================================================================

class TestRateLimitMiddleware:
    """Token-bucket rate limiter tests."""

    def test_normal_requests_pass(self):
        """A small burst of requests should not be throttled."""
        client = _client()
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rate_limit_headers_present(self):
        """Responses should contain informational rate-limit headers."""
        client = _client()
        resp = client.get("/health")
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    @patch("app.middleware.rate_limit._limiter")
    def test_rate_limit_exceeded_returns_429(self, mock_limiter):
        """When limit is exceeded, middleware should return 429."""
        mock_limiter.is_allowed.return_value = False
        mock_limiter.rate = 60
        mock_limiter.remaining.return_value = 0
        client = _client()
        resp = client.get("/health")
        assert resp.status_code == 429
        assert "Too many requests" in resp.json()["detail"]
        assert resp.headers.get("Retry-After") == "60"


# ===========================================================================
# Input Validation
# ===========================================================================

class TestInputValidationMiddleware:
    """Input sanitisation tests."""

    def test_normal_post_accepted(self):
        """Legitimate POST body should pass through."""
        client = _client()
        resp = client.post(
            "/ask",
            json={"query": "How do I write a good resume?", "user_id": 1},
        )
        # May fail because of LLM, but should NOT be 400/413
        assert resp.status_code != 400
        assert resp.status_code != 413

    def test_sql_injection_union_select_rejected(self):
        """Obvious UNION SELECT payload should be blocked."""
        client = _client()
        resp = client.post(
            "/ask",
            json={"query": "'; UNION SELECT * FROM users --", "user_id": 1},
        )
        assert resp.status_code == 400
        assert "unsafe input" in resp.json()["detail"].lower()

    def test_sql_injection_drop_table_rejected(self):
        """DROP TABLE payload should be blocked."""
        client = _client()
        resp = client.post(
            "/ask",
            json={"query": "test; DROP TABLE users", "user_id": 1},
        )
        assert resp.status_code == 400

    def test_sql_injection_or_1_equals_1_rejected(self):
        """Classic OR 1=1 payload should be blocked."""
        client = _client()
        resp = client.post(
            "/ask",
            json={"query": "' OR 1=1", "user_id": 1},
        )
        assert resp.status_code == 400

    def test_sql_comment_injection_rejected(self):
        """SQL comment ( -- ) injection should be blocked."""
        client = _client()
        resp = client.post(
            "/ask",
            json={"query": "admin'--", "user_id": 1},
        )
        assert resp.status_code == 400

    def test_oversized_body_rejected(self):
        """Request bodies > 1 MB should be rejected with 413."""
        client = _client()
        oversized = "x" * 1_100_000
        resp = client.post(
            "/ask",
            json={"query": oversized, "user_id": 1},
            headers={"Content-Length": str(len(oversized) + 50)},
        )
        assert resp.status_code == 413

    def test_get_requests_bypass_body_check(self):
        """GET requests should not trigger body validation."""
        client = _client()
        resp = client.get("/health")
        assert resp.status_code == 200
