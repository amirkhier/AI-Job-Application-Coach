"""
API-key authentication middleware.

When ``API_KEY`` is set in the environment, every request (except public
paths) must include a matching ``X-API-Key`` header or ``api_key`` query
parameter.  When the variable is unset (the default during development),
authentication is bypassed entirely — zero friction.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

# Paths that never require an API key
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Gate requests behind a shared API key when configured."""

    async def dispatch(self, request: Request, call_next):
        # No key configured → dev mode, skip auth entirely
        if settings.API_KEY is None:
            return await call_next(request)

        # Public paths are always accessible
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if key != settings.API_KEY:
            logger.warning(
                "Unauthorized request to %s from %s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        return await call_next(request)
