# TripPilot Backend

FastAPI + AutoGen multi-agent vacation planner.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY
python main.py        # http://localhost:8080
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `MONGO_URI` | ❌ | MongoDB connection (uses in-memory if not set) |
| `GOOGLE_OAUTH_CLIENT_JSON_PATH` | ❌ | Google Calendar OAuth client |
| `GOOGLE_OAUTH_TOKEN_JSON_PATH` | ❌ | Google Calendar OAuth token |
| `PHOENIX_GRPC_ENDPOINT` | ❌ | Phoenix tracing endpoint |
| `TEMPO_ENDPOINT` | ❌ | Grafana Tempo endpoint |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/login` | Get JWT token |
| `POST` | `/sessions` | Start planning session |
| `GET` | `/sessions/{id}/events` | SSE stream for real-time updates |
| `POST` | `/sessions/{id}/messages` | Send follow-up message |
| `POST` | `/sessions/{id}/select` | Select a vacation option |
| `POST` | `/sessions/{id}/confirm` | Confirm booking + calendar |

## Project Structure

```
backend/
├── main.py           # FastAPI app + endpoints
├── utils/
│   ├── config.py     # Environment settings
│   ├── logging_config.py  # Logging setup
│   └── observability.py   # OpenTelemetry tracing
├── agent/
│   ├── agents.py     # Multi-agent orchestration
│   ├── prompts.py    # System prompts
│   └── schemas.py    # JSON envelope models
├── tools/
│   └── tools.py      # Search, booking, calendar tools
├── policy/
│   └── guardrails.py # Safety validations
└── memory/
    └── short_term.py # Session storage
```

## Agent Architecture

```
SelectorGroupChat (Router)
    │
    ├── ConversationAgent  → Clarify requirements, present results
    │
    └── SearchAgent        → Call search_vacation_options tool
```

**Flow:**
1. User sends message → Router decides which agent
2. Missing info? → ConversationAgent asks questions
3. Have all info? → SearchAgent searches → ConversationAgent presents
4. User selects → BookingAgent generates calendar copy

## Google Calendar Setup

1. Create project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable "Google Calendar API"
3. Create OAuth Desktop credentials → Download JSON
4. Run setup:
   ```bash
   python -c "from tools.tools import google_oauth_setup; google_oauth_setup()"
   ```

## Security

- JWT authentication required for all endpoints
- Booking links generated server-side only
- Calendar writes require explicit confirmation
- Guardrails validate all dates and actions