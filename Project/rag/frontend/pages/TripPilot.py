import os
import json
import time
import logging
import threading
import queue
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(
    page_title="TripPilot Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BACKEND_URI = os.getenv("BACKEND_URI", "http://18.222.92.149:8080").rstrip("/")
SESSIONS_URI = f"{BACKEND_URI}/sessions"
TIMEOUT = httpx.Timeout(30.0)


_STATE_DEFAULTS: Dict[str, Any] = {
    "trip_session_id": None,
    "trip_status": "idle",
    "trip_events": [],
    "trip_constraints": {},
    "trip_recommendations": [],
    "trip_selected_option": None,
    "trip_booking_links": None,
    "trip_calendar_event": None,
    "trip_clarify_questions": [],
    "trip_clarify_missing": [],
    "trip_listener": None,
    "trip_messages": [],
    "trip_thinking": False,
    "trip_booking_copy": None,
}


def _init_state(force: bool = False) -> None:
    for key, val in _STATE_DEFAULTS.items():
        if force or key not in st.session_state:
            st.session_state[key] = deepcopy(val)
    if force or "trip_event_queue" not in st.session_state:
        st.session_state["trip_event_queue"] = queue.Queue()


def _reset_conversation() -> None:
    _stop_stream_listener()
    _init_state(force=True)


def _append_user_message(content: str) -> None:
    st.session_state.setdefault("trip_messages", [])
    msgs = st.session_state["trip_messages"]
    # Avoid duplicating same user message if it just was appended
    if msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == content:
        return
    msgs.append({"role": "user", "content": content})


def _clean_agent_response(text: str) -> str:
    """Remove JSON_START...JSON_END blocks and clean up agent response for display."""
    import re
    # Remove everything from JSON_START to JSON_END (handles nested braces)
    cleaned = re.sub(
        r"JSON_START.*?JSON_END",
        "",
        text,
        flags=re.DOTALL
    )
    # Also remove any standalone JSON_START or JSON_END that might remain
    cleaned = cleaned.replace("JSON_START", "").replace("JSON_END", "")
    # Clean up extra whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def _append_assistant_message(content: str) -> None:
    st.session_state.setdefault("trip_messages", [])
    msgs = st.session_state["trip_messages"]
    # Avoid duplicate consecutive assistant messages with same content
    if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("content") == content:
        return
    msgs.append({"role": "assistant", "content": content})


class SSEListener(threading.Thread):
    def __init__(
        self,
        url: str,
        sink: "queue.Queue[Tuple[str, Dict[str, Any]]]",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._url = url
        self._sink = sink
        self._stop_event = threading.Event()
        self._headers = headers or {}
        self._client = httpx.Client(timeout=None)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._client.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def run(self) -> None:  # pragma: no cover - threading logic
        try:
            with self._client.stream("GET", self._url, headers=self._headers) as response:
                event_name: Optional[str] = None
                data_lines: List[str] = []
                for line in response.iter_lines():
                    if self._stop_event.is_set():
                        break
                    if line is None:
                        continue
                    line = line.strip()
                    if not line:
                        if data_lines:
                            raw = "\n".join(data_lines)
                            try:
                                payload = json.loads(raw)
                            except json.JSONDecodeError:
                                payload = {"raw": raw}
                            self._sink.put((event_name or "message", payload))
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith("event:"):
                        event_name = line.split("event:", 1)[1].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.split("data:", 1)[1].strip())
        except Exception as exc:
            self._sink.put(("error", {"message": str(exc)}))


def _stop_stream_listener() -> None:
    listener = st.session_state.get("trip_listener")
    if listener and listener.is_alive():
        listener.stop()
        listener.join(timeout=1.0)
    st.session_state["trip_listener"] = None


def _start_stream_listener(session_id: str) -> None:
    _stop_stream_listener()
    st.session_state["trip_event_queue"] = queue.Queue()
    url = f"{SESSIONS_URI}/{session_id}/events"
    listener = SSEListener(
        url=url,
        sink=st.session_state["trip_event_queue"],
        headers=_auth_headers(),
    )
    listener.start()
    st.session_state["trip_listener"] = listener


def _process_event(event_name: str, payload: Dict[str, Any]) -> None:
    st.session_state["trip_events"].append({
        "ts": time.time(),
        "event": event_name,
        "payload": payload,
    })

    if event_name == "status":
        st.session_state["trip_status"] = payload.get("status", st.session_state["trip_status"])
        if payload.get("status") in {"running", "awaiting_user"}:
            st.session_state["trip_thinking"] = payload.get("status") == "running"
        return

    if event_name == "delta":
        # Accumulate streaming chunks - only show final ConversationAgent output
        text = payload.get("text", "")
        if not text:
            return
        st.session_state.setdefault("trip_stream_buffer", "")
        st.session_state["trip_stream_buffer"] += text
        # Don't update UI on every delta - wait for full message
        return

    if event_name == "message":
        # Full message from an agent
        agent = payload.get("agent", "")
        
        # Clear stream buffer when any agent message arrives
        st.session_state.pop("trip_stream_buffer", None)
        
        # Don't display raw message - we use the structured envelope to generate
        # a cleaner response. Just update thinking state.
        if agent == "ConversationAgent":
            st.session_state["trip_thinking"] = False
        return

    if event_name == "structured":
        # Generate display message from structured envelope
        env = payload.get("data") or {}
        env_type = env.get("type")
        if env_type == "clarify":
            questions = env.get("questions", [])
            st.session_state["trip_clarify_questions"] = questions
            st.session_state["trip_clarify_missing"] = env.get("missing_fields", [])
            if env.get("updated_constraints"):
                st.session_state["trip_constraints"] = env.get("updated_constraints")
            # Generate friendly message from questions
            if questions:
                msg = "I'd love to help plan your trip! I just need a few more details:\n\n"
                for i, q in enumerate(questions, 1):
                    msg += f"{i}. {q}\n"
                _append_assistant_message(msg)
            st.session_state["trip_thinking"] = False
        elif env_type == "recommendations":
            options = env.get("options", [])
            st.session_state["trip_recommendations"] = options
            st.session_state["trip_clarify_questions"] = []
            st.session_state["trip_clarify_missing"] = []
            if env.get("updated_constraints"):
                st.session_state["trip_constraints"] = env.get("updated_constraints")
            # Generate friendly message with top options
            if options:
                msg = f"Great news! I found {len(options)} vacation options for you. Here are the top picks:\n\n"
                for i, opt in enumerate(options[:3], 1):
                    msg += f"**{i}. {opt.get('destination')}** - ${opt.get('est_total_usd')}\n"
                msg += "\nCheck out all options below and click 'Select this option' to proceed with booking!"
                _append_assistant_message(msg)
            st.session_state["trip_thinking"] = False
        elif env_type == "booking_copy":
            st.session_state["trip_booking_copy"] = env
        return

    if event_name in {"error", "structured_error"}:
        # Show user-friendly message instead of technical error
        _append_assistant_message("I'm having trouble processing your request. Please try rephrasing or provide more details about your trip.")
        st.session_state["trip_thinking"] = False
        st.session_state["trip_status"] = "awaiting_user"
        return

    if event_name == "selected":
        st.session_state["trip_selected_option"] = payload.get("selected_option")
    elif event_name == "booking_links":
        st.session_state["trip_booking_links"] = payload
    elif event_name == "calendar_event":
        st.session_state["trip_calendar_event"] = payload


def _drain_event_queue() -> None:
    q = st.session_state.get("trip_event_queue")
    if not q:
        return
    while True:
        try:
            event_name, payload = q.get_nowait()
        except queue.Empty:
            break
        _process_event(event_name, payload)


def _require_token() -> str:
    token = st.session_state.get("jwt_token")
    if not token:
        st.warning("Please log in from the Home page.")
        st.stop()
    return token


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_require_token()}"}


def _start_session(initial_prompt: str) -> None:
    text = initial_prompt.strip()
    if not text:
        st.error("Please enter a prompt to start planning.")
        return
    try:
        resp = httpx.post(
            SESSIONS_URI,
            json={"text": text},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Failed to start session: {exc}")
        return

    session_id = resp.json().get("session_id")
    if not session_id:
        st.error("Backend did not return a session_id.")
        return

    st.session_state.update({
        "trip_session_id": session_id,
        "trip_status": "running",
        "trip_events": [],
        "trip_constraints": {},
        "trip_recommendations": [],
        "trip_selected_option": None,
        "trip_booking_links": None,
        "trip_calendar_event": None,
        "trip_clarify_questions": [],
        "trip_clarify_missing": [],
        "trip_booking_copy": None,
    })
    _start_stream_listener(session_id)


def _send_user_message(message: str) -> None:
    session_id = st.session_state.get("trip_session_id")
    if not session_id:
        st.warning("Start a session first.")
        return
    text = message.strip()
    if not text:
        st.warning("Enter a message before sending.")
        return
    try:
        url = f"{SESSIONS_URI}/{session_id}/messages"
        resp = httpx.post(url, json={"text": text}, headers=_auth_headers(), timeout=TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Failed to send message: {exc}")


def _select_option(index: int) -> None:
    session_id = st.session_state.get("trip_session_id")
    if session_id is None:
        return
    try:
        url = f"{SESSIONS_URI}/{session_id}/select"
        resp = httpx.post(
            url,
            json={"option_index": index},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        # Set selected option immediately for UI feedback
        if result.get("selected_option"):
            st.session_state["trip_selected_option"] = result["selected_option"]
            _append_assistant_message(f"Great choice! You selected **{result['selected_option'].get('destination')}**. Click 'Confirm booking' below to generate booking links and add to your calendar.")
    except httpx.HTTPError as exc:
        st.error(f"Failed to select option: {exc}")


def _confirm_selection(confirm: bool) -> None:
    session_id = st.session_state.get("trip_session_id")
    if session_id is None:
        return
    try:
        url = f"{SESSIONS_URI}/{session_id}/confirm"
        if confirm:
            st.session_state["trip_thinking"] = True
            _append_assistant_message("Generating your booking links and adding to calendar... Please wait.")
        resp = httpx.post(
            url,
            json={"confirm": confirm},
            headers=_auth_headers(),
            timeout=30,  # Longer timeout for calendar creation
        )
        resp.raise_for_status()
        if confirm:
            _append_assistant_message("✅ Done! Your booking links are ready and the trip has been added to your calendar.")
            st.session_state["trip_thinking"] = False
        else:
            _append_assistant_message("No problem! Let me know what you'd like to change about the options.")
    except httpx.HTTPError as exc:
        st.session_state["trip_thinking"] = False
        st.error(f"Failed to confirm: {exc}")


def _render_recommendations() -> None:
    recs = st.session_state.get("trip_recommendations", [])
    if not recs:
        return  # Don't show placeholder - just wait for actual recommendations

    st.subheader("Vacation options")
    for idx, option in enumerate(recs):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"### {idx + 1}. {option.get('destination')} — {option.get('location')}")
            st.write(option.get("why_fits", ""))
            st.write(
                f"**Dates:** {option.get('arrival_date')} → {option.get('leave_date')}  \
                **Est. total:** ${option.get('est_total_usd')} USD"
            )
            highlights = option.get("highlights") or []
            if highlights:
                st.write("**Highlights:** " + ", ".join(highlights))
        with cols[1]:
            st.metric("Budget", f"${option.get('est_total_usd')}")
            st.button(
                "Select this option",
                key=f"select_option_{idx}",
                on_click=_select_option,
                args=(idx,),
            )


def _render_selection_panel() -> None:
    selected = st.session_state.get("trip_selected_option")
    if not selected:
        return
    
    st.subheader("✅ Your Selected Trip")
    
    # Nice card-like display
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### {selected.get('destination')} — {selected.get('location')}")
        st.markdown(f"**📅 Dates:** {selected.get('arrival_date')} → {selected.get('leave_date')}")
        st.markdown(f"**💰 Estimated Total:** ${selected.get('est_total_usd'):,.2f} USD")
        
        highlights = selected.get("highlights") or []
        if highlights:
            st.markdown(f"**✨ Highlights:** {', '.join(highlights)}")
        
        st.markdown(f"*{selected.get('why_fits', '')}*")
        
        # Show booking links
        offer_refs = selected.get("offer_refs") or {}
        if offer_refs.get("flight_url") or offer_refs.get("hotel_url"):
            links = []
            if offer_refs.get("flight_url"):
                links.append(f"[🛫 Search Flights]({offer_refs['flight_url']})")
            if offer_refs.get("hotel_url"):
                links.append(f"[🏨 Search Hotels]({offer_refs['hotel_url']})")
            st.markdown(" | ".join(links))
    
    with col2:
        st.metric("Total", f"${selected.get('est_total_usd'):,.2f}")
    
    st.info("Click **Confirm booking** to generate your booking links and add this trip to your calendar.")
    cols = st.columns(2)
    cols[0].button("Confirm booking", on_click=_confirm_selection, args=(True,), type="primary")
    cols[1].button("Ask for changes", on_click=_confirm_selection, args=(False,), type="secondary")


def _render_booking_artifacts() -> None:
    links = st.session_state.get("trip_booking_links")
    if links:
        st.subheader("Booking links")
        for link in links.get("links", []):
            st.write(f"[{link.get('label')}]({link.get('short_url')}) — expires {time.ctime(link.get('expires_at', 0))}")
        if links.get("disclaimer"):
            st.caption(links["disclaimer"])

    calendar_event = st.session_state.get("trip_calendar_event")
    if calendar_event:
        st.subheader("📅 Calendar Event Added!")
        if calendar_event.get("status") == "ok":
            st.success("Your trip has been added to your Google Calendar!")
            html_link = calendar_event.get("htmlLink")
            if html_link:
                st.markdown(f"[🔗 Open in Google Calendar]({html_link})")
        else:
            st.warning("Calendar event could not be created. You can add it manually.")


def _render_event_log() -> None:
    with st.expander("Event stream", expanded=False):
        for item in reversed(st.session_state.get("trip_events", [])):
            stamp = time.strftime("%H:%M:%S", time.localtime(item["ts"]))
            st.markdown(f"`{stamp}` **{item['event']}** → {json.dumps(item['payload'], ensure_ascii=False)[:400]}")


def _handle_user_prompt(prompt: str) -> None:
    text = prompt.strip()
    if not text:
        return
    _append_user_message(text)
    st.session_state["trip_thinking"] = True
    if st.session_state.get("trip_session_id"):
        _send_user_message(text)
    else:
        _start_session(text)


def main() -> None:
    _init_state()
    _drain_event_queue()

    session_id = st.session_state.get("trip_session_id")
    status = st.session_state.get("trip_status", "idle")

    st.title("TripPilot Conversation")
    top_cols = st.columns([1, 3, 2])
    with top_cols[0]:
        if st.button("New Chat", use_container_width=True):
            _reset_conversation()
            st.rerun()
    with top_cols[1]:
        st.caption(f"Backend: {BACKEND_URI}")
    with top_cols[2]:
        if session_id:
            st.caption(f"Session: {session_id[:8]}… — status: {status}")
        else:
            st.caption("No active session")

    for msg in st.session_state.get("trip_messages", []):
        role = "assistant" if msg.get("role") == "assistant" else "user"
        with st.chat_message(role):
            st.write(msg.get("content", ""))

    if st.session_state.get("trip_thinking"):
        with st.chat_message("assistant"):
            st.write("Thinking…")
        # Show live event stream while generating
        _render_event_log()
        # Auto-refresh to poll for new events while thinking
        time.sleep(0.5)
        st.rerun()

    prompt = st.chat_input("Describe your ideal trip or answer follow-ups")
    if prompt:
        _handle_user_prompt(prompt)
        st.rerun()

    _render_recommendations()
    _render_selection_panel()
    _render_booking_artifacts()


if __name__ == "__main__":
    main()
