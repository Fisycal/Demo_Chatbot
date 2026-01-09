from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict

@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    needs_user_confirm: bool = False
    reason: str = ""

ALLOWED_ACTIONS = {
    "search_vacation_options",
    "create_clickout_links",
    "create_google_calendar_event",
}

def validate_date_range(start: str, end: str) -> None:
    # expects YYYY-MM-DD
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if e <= s:
        raise ValueError("end_date must be after start_date")
    if (e - s).days > 30:
        raise ValueError("date range too long (max 30 days)")

def decide(action: str, payload: Dict[str, Any]) -> GuardrailDecision:
    if action not in ALLOWED_ACTIONS:
        return GuardrailDecision(False, reason=f"action not allowlisted: {action}")

    if action == "search_vacation_options":
        # ensure date range present
        start = payload.get("start_date")
        end = payload.get("end_date")
        if start and end:
            validate_date_range(start, end)
        return GuardrailDecision(True)

    if action == "create_clickout_links":
        # requires user confirmation because it generates external checkout links (but not payment itself)
        return GuardrailDecision(True, needs_user_confirm=True, reason="Generating booking links requires user confirmation")

    if action == "create_google_calendar_event":
        return GuardrailDecision(True, needs_user_confirm=True, reason="Creating a calendar event requires user confirmation")

    return GuardrailDecision(False, reason="unhandled action")
