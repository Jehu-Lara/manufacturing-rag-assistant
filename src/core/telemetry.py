from __future__ import annotations

import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

_SERVICE_NAME = "rag4-api"
_configured = False


def configure_tracing(app: "FastAPI") -> None:
    """Sets up a real TracerProvider so spans created via
    opentelemetry.trace.get_tracer(__name__) actually populate
    JsonFormatter's trace_id field (see src/core/logging.py). Exporting to
    a real collector is optional: OTEL_EXPORTER_OTLP_ENDPOINT unset means
    spans are still created (trace_id still populates logs) but never
    leave the process — a local no-op, not a broken setup."""
    global _configured
    if _configured:
        return
    _configured = True

    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_SERVICE_NAME)
