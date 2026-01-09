from __future__ import annotations

import asyncio
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import uvicorn
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils.config import settings
from utils.observability import setup_tracing
from utils.logging_config import setup_logging
from memory.short_term import EventBus, get_short_term_store
from agent.agents import VacationAgentService
from tools.tools import create_clickout_links, create_google_calendar_event
from policy.guardrails import decide, validate_date_range

logger = setup_logging()

app = FastAPI(title="Vacation Planner Agent (AutoGen, multi-agent)")

# --- Prometheus Metrics ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=False,  # Show 404, 500 separately
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    logger.info("[Metrics] Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("[Metrics] prometheus-fastapi-instrumentator not installed")

store = get_short_term_store()
bus = EventBus()
svc = VacationAgentService(logger=logger)

# --------------- Schemas ---------------
class StartSessionReq(BaseModel):
    text: str = Field(..., description="User prompt, e.g., 'Plan a vacation between 2026-03-10 and 2026-03-17'")


class StartSessionResp(BaseModel):
    session_id: str


class MessageReq(BaseModel):
    text: str


class SelectReq(BaseModel):
    option_index: int


class ConfirmReq(BaseModel):
    confirm: bool = True


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="Access password")


class LoginResponse(BaseModel):
    access_token: str
    email: str
    message: str

# JWT Configuration
JWT_EXPIRATION_HOURS = 24

# Authentication Configuration
client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")
access_password = os.getenv("ACCESS_PASSWORD", "password")
# --- Authentication Functions ---
def authenticate(auth_token: Any) -> Optional[str]:
    """Authenticate user from JWT token and return email (personId)"""
    if not auth_token:
        return None
    try:
        bearer_token: str = auth_token.replace("Bearer ", "")
        output_payload: Dict[str, Any] = jwt.decode(bearer_token, client_secret, algorithms=["HS256"])
        
        if "personId" in output_payload:
            return str(output_payload["personId"])
        
        return None
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
# --- Authentication Endpoint ---
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint that validates credentials and returns JWT token"""
    try:
        # Validate email format
        if not request.email or "@" not in request.email:
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Check if email is in admin list
        if not _is_admin(request.email):
            raise HTTPException(status_code=403, detail="Email not authorized")
        
        # Verify password
        if request.password != access_password:
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Create JWT token with email as personId
        expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS)
        payload = {
            "personId": request.email.lower(),
            "email": request.email.lower(),
            "exp": expiration,
            "iat": datetime.datetime.utcnow()
        }
        
        token = jwt.encode(payload, client_secret, algorithm="HS256")
        
        logger.info(f"User {request.email} logged in successfully")
        
        return LoginResponse(
            access_token=token,
            email=request.email.lower(),
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

def _load_admin_emails() -> List[str]:
    """Load admin emails from admin.txt located either near backend or frontend."""
    candidates = [
        Path(__file__).parent / "admin.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return [line.strip().lower() for line in f if line.strip()]
        except Exception:
            continue
    return []


def _is_admin(email: Optional[str]) -> bool:
    if not email:
        return False
    admins = _load_admin_emails()
    return email.lower() in set(admins)


def require_user(authorization: Optional[str] = Header(default=None)) -> str:
    user = authenticate(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="invalid or missing token")
    return user

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# --------------- Helpers ---------------
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _validate_constraints(constraints: Dict[str, Any]) -> None:
    start = constraints.get("start_date")
    end = constraints.get("end_date")
    if start and end:
        validate_date_range(start, end)

async def _run_team_background(session_id: str, user_text: str) -> None:
    try:
        sess = await store.get_session(session_id)
    except KeyError:
        logger.error("session not found in background run: %s", session_id)
        return

    sess.last_error = None
    await store.save_session(sess)

    await store.append_message(session_id, "user", user_text)
    await bus.publish(session_id, {"event": "status", "data": {"status": "running"}})

    try:
        async for ev in svc.stream_plan_turn(
            session_id=session_id,
            session_constraints=sess.constraints,
            user_text=user_text,
        ):
            await bus.publish(session_id, {"event": ev.get("event", "message"), "data": ev})

            if ev.get("event") == "structured":
                envelope = ev.get("data") or {}
                typ = envelope.get("type")

                updated = envelope.get("updated_constraints")
                if isinstance(updated, dict):
                    try:
                        _validate_constraints(updated)
                    except ValueError as err:
                        logger.warning("invalid constraint update for session %s: %s", session_id, err)
                        await bus.publish(session_id, {"event": "structured_error", "data": {"message": str(err)}})
                    else:
                        sess.constraints = updated
                        sess.last_error = None

                if typ == "recommendations":
                    sess.last_recommendations = envelope.get("options", []) or []
                    sess.status = "awaiting_user"
                    sess.last_error = None
                elif typ == "clarify":
                    sess.status = "awaiting_user"
                    sess.last_error = None

                await store.save_session(sess)

        await bus.publish(session_id, {"event": "status", "data": {"status": "idle"}})
    except Exception as e:
        logger.exception("agent run failed for session=%s: %s", session_id, e)
        await bus.publish(session_id, {"event": "error", "data": {"message": str(e)}})
        sess.status = "error"
        sess.last_error = str(e)
        await store.save_session(sess)

@app.on_event("startup")
async def _startup() -> None:
    setup_tracing(app=app)

@app.post("/sessions", response_model=StartSessionResp)
async def start_session(
    req: StartSessionReq,
    current_user: str = Depends(require_user),
) -> StartSessionResp:
    session_id = await store.create_session()
    asyncio.create_task(_run_team_background(session_id, req.text))
    return StartSessionResp(session_id=session_id)

@app.get("/sessions/{session_id}/events")
async def stream_events(
    session_id: str,
    current_user: str = Depends(require_user),
):
    async def gen():
        yield _sse("status", {"status": "connected", "ts": int(time.time())})
        try:
            async for ev in bus.subscribe(session_id):
                yield _sse(ev.get("event", "message"), ev.get("data", {}))
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("SSE stream error for session=%s: %s", session_id, e)
            return

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    req: MessageReq,
    current_user: str = Depends(require_user),
):
    try:
        _ = await store.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    asyncio.create_task(_run_team_background(session_id, req.text))
    return {"status": "accepted"}

@app.post("/sessions/{session_id}/select")
async def select_option(
    session_id: str,
    req: SelectReq,
    current_user: str = Depends(require_user),
):
    sess = await store.get_session(session_id)
    if not sess.last_recommendations:
        raise HTTPException(status_code=400, detail="no recommendations to select from")
    if req.option_index < 0 or req.option_index >= len(sess.last_recommendations):
        raise HTTPException(status_code=400, detail="option_index out of range")

    selected = sess.last_recommendations[req.option_index]
    sess.selected_option = selected
    sess.status = "awaiting_confirm"
    sess.last_error = None
    await store.save_session(sess)

    await bus.publish(session_id, {"event": "selected", "data": {"selected_option": selected}})
    await bus.publish(session_id, {"event": "needs_confirmation", "data": {
        "action": "confirm_booking_and_calendar",
        "reason": "Confirm to generate booking links and add a tentative vacation event to your Google Calendar.",
        "next_endpoint": f"/sessions/{session_id}/confirm"
    }})
    return {"status": "ok", "selected_option": selected}

@app.post("/sessions/{session_id}/confirm")
async def confirm(
    session_id: str,
    req: ConfirmReq,
    current_user: str = Depends(require_user),
):
    sess = await store.get_session(session_id)
    if not req.confirm:
        sess.status = "awaiting_user"
        sess.last_error = None
        await store.save_session(sess)
        await bus.publish(session_id, {"event": "status", "data": {"status": "awaiting_user", "message": "Okay—tell me what to change and I’ll revise options."}})
        return {"status": "declined"}

    if not sess.selected_option:
        raise HTTPException(status_code=400, detail="no selected option")

    # Guardrails: link generation must be explicitly confirmed
    decision = decide("create_clickout_links", sess.selected_option)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    as_of_ts = int(time.time())
    links = await create_clickout_links(option=sess.selected_option, as_of_ts=as_of_ts)
    sess.booking_links = links
    sess.last_error = None
    await store.save_session(sess)
    await bus.publish(session_id, {"event": "booking_links", "data": links})

    # BookingAgent: generate calendar copy (stream)
    booking_copy = None
    async for ev in svc.stream_booking_copy(
        selected_option=sess.selected_option,
        booking_links=links,
    ):
        await bus.publish(session_id, {"event": ev.get("event", "message"), "data": ev})
        if ev.get("event") == "structured":
            booking_copy = ev.get("data")

    # Calendar write (optional)
    opt = sess.selected_option
    title = (booking_copy or {}).get("title") or f"Vacation: {opt.get('destination')}"
    description = (booking_copy or {}).get("description") or _fallback_description(opt, links)

    cal = await create_google_calendar_event(
        title=title,
        start_date=opt.get("arrival_date"),
        end_date=opt.get("leave_date"),
        description=description,
        location=opt.get("location", ""),
    )
    sess.calendar_event = cal
    sess.status = "completed"
    sess.last_error = None
    await store.save_session(sess)
    await bus.publish(session_id, {"event": "calendar_event", "data": cal})
    await bus.publish(session_id, {"event": "status", "data": {"status": "completed"}})

    return {"status": "ok", "booking_links": links, "calendar_event": cal}

def _fallback_description(opt: Dict[str, Any], links: Dict[str, Any]) -> str:
    lines = [
        f"Destination: {opt.get('destination')} ({opt.get('location')})",
        f"Dates: {opt.get('arrival_date')} to {opt.get('leave_date')}",
        f"Estimated total: ${opt.get('est_total_usd')} USD (prices may change)",
        "",
        "Booking links (click-out):",
    ]
    for l in (links.get("links") or []):
        lines.append(f"{l.get('label')}: {l.get('short_url')}")
    lines.append("")
    lines.append(links.get("disclaimer", ""))
    return "\n".join(lines)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, timeout_keep_alive=60)
