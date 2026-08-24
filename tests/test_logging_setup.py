from __future__ import annotations

import io
import json
import logging

from api.logging_setup import JsonFormatter


def test_json_formatter_emits_expected_keys_and_extra_fields() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_logging_setup.hermetic")
    logger.setLevel("INFO")
    logger.handlers = [handler]
    logger.propagate = False

    logger.info(
        "query received",
        extra={"request_id": "abc-123", "event": "query_received", "latency_ms": 42},
    )

    line = stream.getvalue().strip()
    payload = json.loads(line)

    assert payload["message"] == "query received"
    assert payload["level"] == "INFO"
    assert "timestamp" in payload and payload["timestamp"]
    assert payload["request_id"] == "abc-123"
    assert payload["event"] == "query_received"
    assert payload["latency_ms"] == 42


def test_json_formatter_output_is_single_line_json() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_logging_setup.single_line")
    logger.setLevel("INFO")
    logger.handlers = [handler]
    logger.propagate = False

    logger.warning("disk usage high", extra={"status": "warn"})

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "WARNING"
    assert payload["status"] == "warn"
