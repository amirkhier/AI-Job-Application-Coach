"""
Input sanitisation middleware — defence-in-depth layer.

Checks request bodies for:
  1. Excessive size (> 1 MB)
  2. Obvious SQL-injection patterns

Note: Parameterised queries in ``DatabaseManager`` are the *primary*
SQL-injection defence.  This middleware catches attacks before they even
reach the business logic.
"""

from __future__ import annotations

import logging
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── SQL injection pattern detection ──────────────────────────────────────
# Patterns are intentionally narrow to avoid false positives on resume /
# cover-letter text that naturally contains words like "select", "update",
# PDF binary artefacts that contain "--", etc.
SQL_INJECTION_PATTERNS = [
    r"(\b(UNION)\b\s+(ALL\s+)?\b(SELECT)\b)",
    r"(;\s*(DROP|DELETE|UPDATE|INSERT)\b)",
    r"(\bOR\b\s+1\s*=\s*1)",
    r'(\bOR\b\s+[\x27"]\s*=\s*[\x27"])',
    r"(/\*.*\*/)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in SQL_INJECTION_PATTERNS]

MAX_BODY_SIZE = 1_000_000  # 1 MB


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Reject excessively large or potentially malicious request bodies."""

    async def dispatch(self, request: Request, call_next):
        # ── Body-size check ──────────────────────────────────────────
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            logger.warning(
                "Request body too large (%s bytes) from %s",
                content_length,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large (max 1 MB)"},
            )

        # ── SQL injection check (POST / PUT / PATCH only) ───────────
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            body_text = body.decode("utf-8", errors="ignore")

            for pattern in _COMPILED:
                if pattern.search(body_text):
                    logger.warning(
                        "Potentially unsafe input detected from %s on %s",
                        request.client.host if request.client else "unknown",
                        request.url.path,
                    )
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Potentially unsafe input detected"},
                    )

            # Re-attach body so downstream handlers can still read it.
            # (Starlette consumes the stream on first read.)
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

        return await call_next(request)
