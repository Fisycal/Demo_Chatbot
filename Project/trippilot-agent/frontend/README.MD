# TripPilot Frontend

Streamlit chat UI for the TripPilot vacation planner.

## Quick Start

```bash
# Mac/Linux
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py

# Windows
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Home.py
```

Opens at http://localhost:8501

## Pages

| Page | Purpose |
|------|---------|
| `Home.py` | Login page |
| `pages/TripPilot.py` | Main vacation planning chat |

## Configuration

Create `.env`:

```bash
BACKEND_URI=http://localhost:8080
ACCESS_PASSWORD=password
```

## How it Works

1. **Login** → Authenticate with email/password
2. **Chat** → Describe your ideal vacation
3. **View options** → See AI-generated recommendations
4. **Select** → Choose an option
5. **Confirm** → Get booking links + calendar event

## Project Structure

```
frontend/
├── Home.py              # Login page
├── pages/
│   └── TripPilot.py     # Main chat interface
├── admin.txt            # Authorized email list
└── requirements.txt
```

## Requirements

- Python 3.10+
- Backend running at `BACKEND_URI`
