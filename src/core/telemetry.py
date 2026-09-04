from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

_SERVICE_NAME = "rag4-api"


def _already_configured() -> bool:
    """Idempotence read off the live provider, not off a module-level bool.
    OpenTelemetry's default is a ProxyTracerProvider; once our own SDK
    TracerProvider carrying our service name is installed, a second call is a
    no-op — and unlike a module flag this stays correct when another entry
    point or an earlier test has already installed one."""
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return False
    return bool(provider.resource.attributes.get("service.name") == _SERVICE_NAME)


def configure_tracing(app: "FastAPI", *, otlp_endpoint: str | None = None) -> None:
    """Sets up a real TracerProvider so spans created via
    opentelemetry.trace.get_tracer(__name__) actually populate
    JsonFormatter's trace_id field (see src/core/logging.py). Exporting to
    a real collector is optional: no endpoint means spans are still created
    (trace_id still populates logs) but never leave the process — a local
    no-op, not a broken setup."""
    if _already_configured():
        return

    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_SERVICE_NAME)
