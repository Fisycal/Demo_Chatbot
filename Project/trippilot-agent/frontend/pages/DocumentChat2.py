import os
import time
import uuid
import json
import httpx
import asyncio
import logging
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="TripPilot – Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# ENV + BACKEND CONFIG
# ---------------------------------------------------------
load_dotenv()
backend_uri = os.getenv("BACKEND_URI", "http://localhost:8080")

SESSIONS_URL = f"{backend_uri}/sessions"
MESSAGES_URL = lambda sid: f"{backend_uri}/sessions/{sid}/messages"
EVENTS_URL = lambda sid: f"{backend_uri}/sessions/{sid}/events"

TIMEOUT = httpx.Timeout(60.0)

# ---------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

if "jwt_token" not in st.session_state:
    st.warning("Please log in from the Home page.")
    st.stop()

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def auth_header():
    return {"Authorization": f"Bearer {st.session_state['jwt_token']}"}

def start_new_session(initial_text: str):
    """Start a new backend session."""
    resp = httpx.post(
        SESSIONS_URL,
        json={"text": initial_text},
        headers=auth_header(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    st.session_state["session_id"] = session_id
    return session_id

async def stream_events(session_id: str, placeholder):
    """Stream SSE events from backend."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", EVENTS_URL(session_id), headers=auth_header()) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line.replace("data: ", ""))
                    msg = payload.get("message") or payload.get("data", {}).get("message")
                    if msg:
                        placeholder.markdown(
                            f"<div style='background-color:#E6F2FF;padding:10px;"
                            f"border-radius:8px;color:#333;width:fit-content;margin-right:auto;'>"
                            f"{msg}</div>",
                            unsafe_allow_html=True,
                        )
                        st.session_state["chat_history"].append(
                            {"sender": "assistant", "message": msg}
                        )
                        st.experimental_rerun()

def send_message(session_id: str, text: str):
    """Send a message to backend."""
    resp = httpx.post(
        MESSAGES_URL(session_id),
        json={"text": text},
        headers=auth_header(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------
def render_chat():
    for entry in st.session_state["chat_history"]:
        if entry["sender"] == "user":
            st.markdown(
                f"<div style='background-color:#4DA6FF;padding:10px;border-radius:8px;"
                f"color:white;width:fit-content;margin-left:auto;'>"
                f"{entry['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background-color:#E6F2FF;padding:10px;border-radius:8px;"
                f"color:#333;width:fit-content;margin-right:auto;'>"
                f"{entry['message']}</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------
def main():
    st.title("TripPilot – Chat With Your AI Travel Planner")

    # NEW CHAT BUTTON
    if st.button("New Chat"):
        st.session_state["chat_history"] = []
        st.session_state["session_id"] = None
        st.experimental_rerun()

    # Show chat history
    render_chat()

    # Chat input
    user_input = st.chat_input("Type your message…")
    if user_input:
        st.session_state["chat_history"].append({"sender": "user", "message": user_input})

        # Start session if needed
        if st.session_state["session_id"] is None:
            session_id = start_new_session(user_input)
        else:
            session_id = st.session_state["session_id"]
            send_message(session_id, user_input)

        # Placeholder for streaming assistant response
        placeholder = st.empty()

        # Run SSE streaming
        asyncio.run(stream_events(session_id, placeholder))

if __name__ == "__main__":
    main()