"""OTEL + Prometheus telemetry wiring — same pattern as other BrighterProject services."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.settings import settings

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "room_classifier_predictions_total",
    "Total number of predictions by room type",
    ["room_type"],
)

PREDICTION_LATENCY = Histogram(
    "room_classifier_prediction_duration_seconds",
    "Time spent processing prediction requests",
    ["endpoint"],
)

# Module-level tracer — populated at startup
_tracer: trace.Tracer | None = None


def setup_telemetry() -> None:
    """Initialize OpenTelemetry tracing and Prometheus metrics."""
    global _tracer

    resource = Resource.create(
        {
            SERVICE_NAME: settings.service_name,
            SERVICE_VERSION: settings.service_version,
        }
    )

    # ---- Tracing ----
    tracer_provider = TracerProvider(resource=resource)

    if not settings.otel_sdk_disabled:
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            )
        )

    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(settings.service_name, settings.service_version)

    # ---- Metrics (Prometheus) ----
    reader = PrometheusMetricReader()
    metrics_provider = MeterProvider(resource=resource, metric_readers=[reader])
    from opentelemetry import metrics as metrics_api

    metrics_api.set_meter_provider(metrics_provider)


def get_tracer() -> trace.Tracer:
    """Return the configured OTEL tracer."""
    if _tracer is None:
        raise RuntimeError("Telemetry has not been initialized. Call setup_telemetry() first.")
    return _tracer


def get_metrics_response() -> tuple[str, str]:
    """Generate the Prometheus metrics endpoint response.

    Returns:
        Tuple of (metrics_body, content_type).
    """
    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
