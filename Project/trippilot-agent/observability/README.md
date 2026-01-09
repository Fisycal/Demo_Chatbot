# Observability Stack

Docker Compose stack for monitoring TripPilot.

## Quick Start

```bash
docker compose up -d
```

## Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Phoenix** | http://localhost:6006 | LLM traces, token usage, agent flow |
| **Grafana** | http://localhost:3000 | Metrics dashboard (login: admin/admin) |
| **Prometheus** | http://localhost:9090 | Metrics storage |
| **Tempo** | http://localhost:3200 | Trace storage |

## What Each Tool Shows

### Phoenix (LLM Observability)
- Full prompt/response for each agent
- Token usage & estimated cost
- Latency per LLM call
- Tool call inputs/outputs

### Grafana (API Metrics)
- Request count & rate
- Error rates (4xx, 5xx)
- Response latency (p50, p95, p99)
- Requests in progress

## Architecture

```
Backend (FastAPI)
    │
    ├── OpenTelemetry ──► Phoenix (port 4317, gRPC)
    │                         └── LLM traces
    │
    ├── OpenTelemetry ──► Tempo (port 4318, HTTP)
    │                         └── HTTP traces
    │
    └── /metrics ──────► Prometheus (scrapes every 15s)
                              └── Grafana (queries)
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service definitions |
| `prometheus.yml` | Prometheus scrape config |
| `tempo.yaml` | Tempo storage config |
| `grafana-datasources.yaml` | Grafana data sources |
| `grafana-dashboard.json` | Pre-built dashboard |

## Commands

```bash
# Start all
docker compose up -d

# View logs
docker compose logs -f phoenix
docker compose logs -f tempo

# Stop all
docker compose down

# Reset (delete data)
docker compose down -v
```

## Backend Configuration

Add to `backend/.env`:

```bash
PHOENIX_GRPC_ENDPOINT=http://localhost:4317
TEMPO_ENDPOINT=http://localhost:4318
PHOENIX_PROJECT_NAME=trip-pilot-agent
```

# Notes on Observability
- Tempo is a distributed tracing backend by Grafana Labs. It stores and queries trace data.
- Phoenix is an LLM observability backend by Arize. It stores and queries LLM traces.

# Tempo answers: "Why is my API slow? Which endpoint failed?"
# Phoenix answers: "What did the LLM say? How many tokens did it use?"