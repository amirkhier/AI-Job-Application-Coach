"""
Structured logging configuration with per-request correlation IDs.

Usage
-----
Call ``setup_logging()`` once during application startup (before any log
calls).  The ``request_id_var`` context variable is set by the
``RequestIdMiddleware`` in *main.py* so that every log line produced
while handling a request carries the same correlation ID.

In **development** the default formatter is human-readable text.
In **production** switch to ``LOG_FORMAT=json`` for structured output
that can be ingested by CloudWatch / ELK / Datadog / etc.
"""

from __future__ import annotations

import copy
import logging
import logging.config
from contextvars import ContextVar

# --------------------------------------------------------------------------- #
#  Per-request correlation ID
# --------------------------------------------------------------------------- #

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


# --------------------------------------------------------------------------- #
#  Filter that injects request_id into every log record
# --------------------------------------------------------------------------- #

class RequestIdFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = request_id_var.get("-")  # type: ignore[attr-defined]
        return True


# --------------------------------------------------------------------------- #
#  Base dict-config template
# --------------------------------------------------------------------------- #

_LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "text": {
            "format": (
                "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s] %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": (
                "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
            ),
        },
    },
    "filters": {
        "request_id": {
            "()": "app.logging_config.RequestIdFilter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "text",  # overridden at bootstrap via setup_logging()
            "filters": ["request_id"],
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
    "loggers": {
        "uvicorn": {"level": "INFO"},
        "uvicorn.access": {"level": "WARNING"},
        "app": {"level": "DEBUG", "propagate": True},
    },
}


# --------------------------------------------------------------------------- #
#  Public bootstrap function
# --------------------------------------------------------------------------- #

def setup_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """Initialise the logging subsystem.

    Parameters
    ----------
    log_level:
        Root log level (``DEBUG``, ``INFO``, ``WARNING``, …).
    log_format:
        ``"text"`` for human-readable output (dev) or ``"json"`` for
        structured JSON (prod).  ``"json"`` requires the
        ``python-json-logger`` package.
    """
    config = copy.deepcopy(_LOGGING_CONFIG)
    config["root"]["level"] = log_level.upper()

    # Select formatter — fall back to text if json logger is not installed
    if log_format == "json":
        try:
            import pythonjsonlogger  # noqa: F401
            config["handlers"]["console"]["formatter"] = "json"
        except ImportError:
            logging.warning(
                "python-json-logger not installed — falling back to text formatter"
            )
            config["handlers"]["console"]["formatter"] = "text"
    else:
        config["handlers"]["console"]["formatter"] = "text"

    logging.config.dictConfig(config)
