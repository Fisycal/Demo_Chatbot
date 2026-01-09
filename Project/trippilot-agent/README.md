# TripPilot 🛫

AI-powered vacation planner using multi-agent architecture.

## What it Does

1. **Chat** → Describe your ideal vacation
2. **Search** → AI finds matching destinations with flights & hotels
3. **Book** → Get booking links and add to Google Calendar

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│                   (Streamlit Chat UI)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP + SSE
┌─────────────────────────▼───────────────────────────────────┐
│                         BACKEND                              │
│                      (FastAPI + AutoGen)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SelectorGroupChat (Router)              │    │
│  │  ┌──────────────────┐    ┌──────────────────┐       │    │
│  │  │ ConversationAgent│    │   SearchAgent    │       │    │
│  │  │  (clarify/present)│    │  (search tool)   │       │    │
│  │  └──────────────────┘    └──────────────────┘       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     OBSERVABILITY                            │
│         Phoenix (LLM traces) + Grafana (metrics)            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Start infrastructure (MongoDB, Phoenix, Grafana)
cd observability
docker compose up -d

# 2. Start backend
cd ../backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
python main.py

# 3. Start frontend
cd ../frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```

## URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:8501 | Chat UI |
| **Backend API** | http://localhost:8080 | REST API |
| **Phoenix** | http://localhost:6006 | LLM traces |
| **Grafana** | http://localhost:3000 | Metrics dashboard |
| **Prometheus** | http://localhost:9090 | Metrics store |

## Project Structure

```
TripPilot-Agent-v2/
├── backend/           # FastAPI + AutoGen agents
├── frontend/          # Streamlit chat UI
└── observability/     # Docker stack (Phoenix, Grafana, Prometheus)
```

See individual READMEs for details:
- [Backend](./backend/README.md) - API, agents, configuration
- [Frontend](./frontend/README.md) - UI setup
- [Observability](./observability/README.md) - Monitoring stack

## Tech Stack

- **Backend**: FastAPI, AutoGen, OpenAI, MongoDB
- **Frontend**: Streamlit
- **Observability**: Phoenix, Grafana, Prometheus, Tempo
- **Infra**: Docker Compose

## License

Internal use only.
