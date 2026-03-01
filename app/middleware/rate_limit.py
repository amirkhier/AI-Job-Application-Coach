"""
In-memory token-bucket rate limiter middleware.

Limits each client IP to a configurable number of requests per window.
Good enough for single-instance deployments; for multi-replica setups,
swap the storage backend to Redis.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Simple token-bucket rate limiter backed by an in-process dict."""

    def __init__(self, rate: int = 60, per: int = 60):
        self.rate = rate  # max requests
        self.per = per    # per N seconds
        self._buckets: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        bucket = self._buckets[key]
        # Purge timestamps older than the current window
        bucket[:] = [t for t in bucket if now - t < self.per]
        if len(bucket) >= self.rate:
            return False
        bucket.append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.time()
        bucket = self._buckets[key]
        bucket[:] = [t for t in bucket if now - t < self.per]
        return max(0, self.rate - len(bucket))


# Module-level limiter instance — shared across all requests
_limiter = InMemoryRateLimiter(
    rate=int(getattr(settings, "RATE_LIMIT_PER_MINUTE", 60)),
    per=60,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed the per-IP rate limit."""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        if not _limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for %s on %s", client_ip, request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests — please slow down"},
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
        # Informational headers (best-effort)
        response.headers["X-RateLimit-Limit"] = str(_limiter.rate)
        response.headers["X-RateLimit-Remaining"] = str(_limiter.remaining(client_ip))
        return response
