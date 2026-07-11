"""Structured logging for the research service.

Every progress event, cost refresh, and lifecycle transition is emitted
as one JSON line on the ``research_service`` logger, so the full event
stream of a run is greppable and machine-parsable after the fact.
"""

from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload)


def get_logger() -> logging.Logger:
    return logging.getLogger("research_service")


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = get_logger()
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_event(kind: str, **data) -> None:
    get_logger().info(kind, extra={"data": {"kind": kind, "ts": time.time(), **data}})
