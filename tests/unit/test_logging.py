import json
import logging

from ai_eos.logging_setup import JsonFormatter, configure_logging, request_id


def test_json_formatter_redacts_and_correlates() -> None:
    request_id.set("req-1")
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    rec.extra_fields = {"api_key": "sk-123", "user": "u1"}
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "hello world" and out["request_id"] == "req-1"
    assert out["api_key"] == "***" and out["user"] == "u1" and "boom" in out["exc"]


def test_configure_logging() -> None:
    configure_logging("debug", json_logs=True)
    assert isinstance(logging.getLogger().handlers[0].formatter, JsonFormatter)
    configure_logging("info", json_logs=False)
    assert logging.getLogger().level == logging.INFO
