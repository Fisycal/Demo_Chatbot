import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import Sampler, SamplingResult, Decision


class LLMAndHTTPSampler(Sampler):
    """Sampler that captures LLM spans and HTTP endpoints, filters AutoGen noise."""
    
    # Block these patterns (AutoGen internal noise)
    BLOCKED_PATTERNS = (
        "autogen",
        "publish",
        "output_topic",
        "group_topic",
    )
    
    def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None):
        name_lower = name.lower() if name else ""
        
        # BLOCK AutoGen internal spans
        for blocked in self.BLOCKED_PATTERNS:
            if blocked in name_lower:
                return SamplingResult(Decision.DROP)
        
        # ALLOW everything else (HTTP, LLM, etc.)
        return SamplingResult(Decision.RECORD_AND_SAMPLE)
    
    def get_description(self):
        return "LLMAndHTTPSampler"


def setup_tracing(service_name: str = "trip-pilot-agent", app=None) -> None:
    """Configure OpenTelemetry tracing for Phoenix (LLM only) and Grafana Tempo (HTTP).
    
    Environment variables:
      - PHOENIX_GRPC_ENDPOINT: Arize Phoenix gRPC (default: http://localhost:4317)
      - TEMPO_ENDPOINT: Grafana Tempo HTTP (default: http://localhost:4318)
      - PHOENIX_PROJECT_NAME: Project name in Phoenix (default: trip-pilot-agent)
    
    Args:
        service_name: Name of the service for tracing
        app: FastAPI app instance for instrumentation
    """
    project_name = os.getenv("PHOENIX_PROJECT_NAME", service_name)
    phoenix_grpc_endpoint = os.getenv("PHOENIX_GRPC_ENDPOINT", "http://18.222.92.149:4317")
    tempo_http_endpoint = os.getenv("TEMPO_ENDPOINT", "http://18.222.92.149:4318")
    
    # Set Phoenix project name via environment (Phoenix reads this)
    os.environ["PHOENIX_PROJECT_NAME"] = project_name
    
    # --- Create TracerProvider with LLM-only sampler ---
    resource = Resource.create({
        "service.name": project_name,
        "service.version": os.getenv("APP_VERSION", "dev"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        "project.name": project_name,  # Standard OTEL attribute
    })
    
    provider = TracerProvider(
        resource=resource,
        sampler=LLMAndHTTPSampler(),  # Filter AutoGen noise, keep HTTP + LLM
    )
    
    # --- Phoenix Exporter (gRPC on port 4317) ---
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcExporter
        
        # Phoenix uses headers to determine project name
        phoenix_headers = {"phoenix-project-name": project_name}
        
        phoenix_exporter = OTLPGrpcExporter(
            endpoint=phoenix_grpc_endpoint,
            insecure=True,
            headers=phoenix_headers,
        )
        provider.add_span_processor(BatchSpanProcessor(phoenix_exporter))
        print(f"[Tracing] Phoenix exporter (gRPC): {phoenix_grpc_endpoint} -> project: {project_name}")
    except ImportError:
        print("[Tracing] OTLP gRPC exporter not installed")
    except Exception as e:
        print(f"[Tracing] Phoenix exporter failed: {e}")
    
    # --- Tempo Exporter (HTTP on port 4318) ---
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpExporter
        
        tempo_exporter = OTLPHttpExporter(
            endpoint=f"{tempo_http_endpoint}/v1/traces",
        )
        provider.add_span_processor(BatchSpanProcessor(tempo_exporter))
        print(f"[Tracing] Tempo exporter (HTTP): {tempo_http_endpoint}/v1/traces")
    except ImportError:
        print("[Tracing] OTLP HTTP exporter not installed")
    except Exception as e:
        print(f"[Tracing] Tempo exporter failed: {e}")
    
    # --- Set global tracer provider ---
    trace.set_tracer_provider(provider)
    
    # --- Instrument FastAPI (for Grafana/Tempo HTTP metrics) ---
    if app:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
            print("[Tracing] FastAPI instrumented")
        except ImportError:
            pass
        except Exception as e:
            print(f"[Tracing] FastAPI instrumentation failed: {e}")
    
    # --- Instrument OpenAI (for Phoenix LLM traces with token usage) ---
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        print("[Tracing] OpenAI instrumented (LLM calls + token usage)")
    except ImportError:
        print("[Tracing] OpenAI instrumentation not installed. Run: pip install openinference-instrumentation-openai")
    except Exception as e:
        print(f"[Tracing] OpenAI instrumentation failed: {e}")
    
    print(f"[Tracing] Setup complete: {project_name} (AutoGen noise filtered, LLM only)")
