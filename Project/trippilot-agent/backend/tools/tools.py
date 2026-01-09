from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.config import settings
from policy.guardrails import decide, GuardrailDecision, validate_date_range

# -------------------------
# Schemas
# -------------------------
class VacationConstraints(BaseModel):
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    departure_city: Optional[str] = None
    destination: Optional[str] = None  # Target country/city if user specified
    budget_total_usd: Optional[float] = None
    travelers: Optional[int] = None
    travelers_count: Optional[int] = None  # Alias
    kids: Optional[int] = None
    vibe: Optional[str] = None  # beach/city/nature
    activities: List[str] = Field(default_factory=list)
    accommodation_type: Optional[str] = None  # hotel/resort/airbnb
    stars_min: Optional[int] = None
    passport_constraints: Optional[str] = None

class VacationOption(BaseModel):
    option_id: str
    destination: str
    location: str
    est_total_usd: float
    arrival_date: str
    leave_date: str
    highlights: List[str]
    why_fits: str
    offer_refs: Dict[str, str]  # e.g. {"flight_offer_id":"...", "hotel_offer_id":"..."}

class VacationOptionsResult(BaseModel):
    as_of_ts: int
    currency: str = "USD"
    options: List[VacationOption]

class ClickoutLink(BaseModel):
    label: str
    short_url: str
    provider: str
    expires_at: int
    price_as_of_ts: int

class ClickoutLinksResult(BaseModel):
    option_id: str
    links: List[ClickoutLink]
    disclaimer: str

# -------------------------
# Deterministic mock provider
# -------------------------
_DESTINATIONS = [
    # Mexico destinations
    ("Cancún", "Quintana Roo, Mexico", "beach", ["all-inclusive", "snorkeling", "cenotes"]),
    ("Playa del Carmen", "Quintana Roo, Mexico", "beach", ["beach clubs", "5th avenue", "cenotes"]),
    ("Tulum", "Quintana Roo, Mexico", "beach", ["ruins", "eco-hotels", "cenotes"]),
    ("Puerto Vallarta", "Jalisco, Mexico", "beach", ["malecón", "whale watching", "old town"]),
    ("Los Cabos", "Baja California Sur, Mexico", "beach", ["luxury resorts", "fishing", "arch"]),
    ("Mexico City", "CDMX, Mexico", "city", ["museums", "tacos", "architecture"]),
    ("Oaxaca", "Oaxaca, Mexico", "city", ["mezcal", "crafts", "ruins"]),
    ("San Miguel de Allende", "Guanajuato, Mexico", "city", ["colonial", "art", "wine"]),
    ("Riviera Maya", "Quintana Roo, Mexico", "beach", ["resorts", "diving", "ruins"]),
    ("Cozumel", "Quintana Roo, Mexico", "beach", ["diving", "cruise port", "reefs"]),
    # Bahamas destinations
    ("Nassau", "New Providence, Bahamas", "beach", ["atlantis", "beaches", "downtown"]),
    ("Paradise Island", "Bahamas", "beach", ["resorts", "aquarium", "casinos"]),
    ("Exuma", "Bahamas", "beach", ["swimming pigs", "cays", "snorkeling"]),
    ("Grand Bahama", "Bahamas", "beach", ["freeport", "nature", "beaches"]),
    # USA destinations
    ("San Diego", "California, USA", "beach", ["beach", "zoo", "sunset cliffs"]),
    ("Miami", "Florida, USA", "beach", ["nightlife", "art deco", "beach"]),
    ("Honolulu", "Hawaii, USA", "beach", ["surfing", "hikes", "luau"]),
    ("Sedona", "Arizona, USA", "nature", ["hiking", "red rocks", "spa"]),
    ("Denver", "Colorado, USA", "nature", ["hiking", "brewery", "Rockies"]),
    # Other international
    ("Vancouver", "British Columbia, Canada", "city", ["food", "mountains", "sea-to-sky"]),
    ("Lisbon", "Portugal", "city", ["tiles", "seafood", "day trips"]),
    ("Barcelona", "Spain", "city", ["architecture", "tapas", "beach nearby"]),
    ("Tokyo", "Japan", "city", ["ramen", "museums", "neighborhoods"]),
    ("Reykjavik", "Iceland", "nature", ["hot springs", "northern lights", "waterfalls"]),
    ("Banff", "Alberta, Canada", "nature", ["lakes", "hikes", "wildlife"]),
]

def _stable_rng(seed_text: str) -> random.Random:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16)
    return random.Random(seed)

def _ensure_constraints(c: VacationConstraints) -> None:
    if not c.start_date or not c.end_date:
        raise ValueError("start_date and end_date are required")
    validate_date_range(c.start_date, c.end_date)

def _dates(c: VacationConstraints) -> Tuple[str, str]:
    _ensure_constraints(c)
    return c.start_date, c.end_date  # type: ignore

def _budget_base(c: VacationConstraints) -> float:
    if c.budget_total_usd and c.budget_total_usd > 0:
        return float(c.budget_total_usd)
    # fallback: estimate based on travelers
    t = c.travelers or 1
    return 1800.0 * t

async def search_vacation_options(constraints: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Search and rank vacation options based on constraints.

    This is a deterministic mock provider. Replace with real travel APIs.
    """
    decision: GuardrailDecision = decide("search_vacation_options", constraints)
    if not decision.allowed:
        return {"status": "denied", "reason": decision.reason}

    c = VacationConstraints(**constraints)
    # Handle travelers_count alias
    if c.travelers_count and not c.travelers:
        c.travelers = c.travelers_count
    start, end = _dates(c)
    rng = _stable_rng(session_id + json.dumps(constraints, sort_keys=True))

    # Filter by destination if user specified a country/region
    candidates = _DESTINATIONS
    if c.destination:
        dest_lower = c.destination.lower()
        # Match destination in the location field (e.g., "Mexico", "Bahamas")
        candidates = [d for d in candidates if dest_lower in d[1].lower() or dest_lower in d[0].lower()]
        if not candidates:
            # Fallback: return all if no match found
            candidates = _DESTINATIONS
    
    # Further filter by vibe if provided
    if c.vibe:
        vibe_filtered = [d for d in candidates if d[2] == c.vibe.lower()]
        if vibe_filtered:
            candidates = vibe_filtered

    # create 10 options
    budget = _budget_base(c)
    travelers = c.travelers or 1
    options: List[VacationOption] = []

    for i, (dest, loc, vibe, tags) in enumerate(rng.sample(candidates, k=min(len(candidates), 10))):
        # synthesize costs
        flight = rng.uniform(250, 900) * travelers
        hotel = rng.uniform(120, 420) * ((date.fromisoformat(end) - date.fromisoformat(start)).days)
        activities = rng.uniform(100, 450) * travelers
        est_total = flight + hotel + activities

        # nudge cost into budget-ish range
        est_total = min(est_total, budget * rng.uniform(0.85, 1.15))

        option_id = f"opt_{hashlib.md5((session_id+dest+str(i)).encode()).hexdigest()[:10]}"
        options.append(VacationOption(
            option_id=option_id,
            destination=dest,
            location=loc,
            est_total_usd=round(est_total, 2),
            arrival_date=start,
            leave_date=end,
            highlights=[*tags[:2], (c.activities[0] if c.activities else tags[-1])],
            why_fits=f"Matches your vibe ({c.vibe or vibe}) and budget signal; great for {', '.join(tags[:2])}.",
            offer_refs={
                "flight_offer_id": f"flt_{hashlib.sha1((option_id+'f').encode()).hexdigest()[:10]}",
                "hotel_offer_id": f"htl_{hashlib.sha1((option_id+'h').encode()).hexdigest()[:10]}",
                # Mock URLs for display (real links generated after confirmation)
                "flight_url": f"https://www.google.com/travel/flights?q={dest.replace(' ', '+')}+flights",
                "hotel_url": f"https://www.google.com/travel/hotels?q={dest.replace(' ', '+')}+hotels",
            }
        ))

    return VacationOptionsResult(as_of_ts=int(time.time()), options=options).model_dump()

async def create_clickout_links(option: Dict[str, Any], as_of_ts: int) -> Dict[str, Any]:
    """Create click-out booking links for the selected option.

    Returns short URLs hosted by your redirector (SHORTLINK_BASE).
    In production, store mapping {token -> provider_url} in your DB and serve redirects.
    """
    decision = decide("create_clickout_links", option)
    if not decision.allowed:
        return {"status": "denied", "reason": decision.reason}
    # NOTE: confirmation is enforced by API layer; this tool is "pure".
    opt = VacationOption(**option)
    now = int(time.time())
    expires = now + 2 * 60 * 60  # 2 hours

    links: List[ClickoutLink] = []
    dest_encoded = opt.destination.replace(" ", "+")
    
    # Use real Google Travel URLs for demo purposes
    # In production, replace with actual booking partner APIs (Amadeus, Booking.com, etc.)
    link_templates = [
        ("Flight", "Google Flights", f"https://www.google.com/travel/flights?q=flights+to+{dest_encoded}"),
        ("Hotel", "Google Hotels", f"https://www.google.com/travel/hotels?q={dest_encoded}+hotels"),
        ("Package", "Google Travel", f"https://www.google.com/travel/explore?q={dest_encoded}+vacation"),
    ]
    
    for kind, provider, url in link_templates:
        links.append(ClickoutLink(
            label=f"Book {kind} ({opt.destination})",
            short_url=url,
            provider=provider,
            expires_at=expires,
            price_as_of_ts=as_of_ts,
        ))
    return ClickoutLinksResult(
        option_id=opt.option_id,
        links=links,
        disclaimer="Prices and availability may change. Links expire; refresh if needed.",
    ).model_dump()

# -------------------------
# Google Calendar tool (optional)
# -------------------------
async def create_google_calendar_event(
    *,
    title: str,
    start_date: str,
    end_date: str,
    description: str,
    location: str = "",
    timezone: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an all-day Google Calendar event for the vacation.

    Requires OAuth configuration:
    - GOOGLE_OAUTH_CLIENT_JSON_PATH (downloaded from Google Cloud Console)
    - GOOGLE_OAUTH_TOKEN_JSON_PATH (created after first auth run)

    For first-time setup, run:
      python -c "from project.tools.tools import google_oauth_setup; google_oauth_setup()"
    """
    decision = decide("create_google_calendar_event", {"start_date": start_date, "end_date": end_date})
    if not decision.allowed:
        return {"status": "denied", "reason": decision.reason}

    if not settings.GOOGLE_OAUTH_CLIENT_JSON_PATH or not settings.GOOGLE_OAUTH_TOKEN_JSON_PATH:
        return {"status": "needs_setup", "reason": "Google OAuth paths not configured"}

    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
    except Exception as e:
        return {"status": "error", "reason": f"missing google deps: {e}"}

    tz = timezone or settings.GOOGLE_TIMEZONE

    try:
        creds = Credentials.from_authorized_user_file(settings.GOOGLE_OAUTH_TOKEN_JSON_PATH, scopes=[
            "https://www.googleapis.com/auth/calendar.events"
        ])
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        event = {
            "summary": title,
            "location": location,
            "description": description,
            # all-day event: end.date is exclusive, so add 1 day to include the last day
            "start": {"date": start_date},
            "end": {"date": (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()},
        }
        created = service.events().insert(calendarId=settings.GOOGLE_CALENDAR_ID, body=event).execute()
        return {
            "status": "ok",
            "event_id": created.get("id"),
            "htmlLink": created.get("htmlLink"),
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}

def google_oauth_setup() -> None:
    """Interactive OAuth setup helper (run locally)."""
    if not settings.GOOGLE_OAUTH_CLIENT_JSON_PATH or not settings.GOOGLE_OAUTH_TOKEN_JSON_PATH:
        raise RuntimeError("Set GOOGLE_OAUTH_CLIENT_JSON_PATH and GOOGLE_OAUTH_TOKEN_JSON_PATH first")
    from google_auth_oauthlib.flow import InstalledAppFlow
    scopes = ["https://www.googleapis.com/auth/calendar.events"]
    flow = InstalledAppFlow.from_client_secrets_file(settings.GOOGLE_OAUTH_CLIENT_JSON_PATH, scopes=scopes)
    creds = flow.run_local_server(port=0)
    with open(settings.GOOGLE_OAUTH_TOKEN_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
