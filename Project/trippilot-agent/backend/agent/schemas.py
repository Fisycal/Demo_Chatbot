from __future__ import annotations

from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, Field, ConfigDict

class BaseEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Allow extra fields from LLM

class ClarifyEnvelope(BaseEnvelope):
    type: Literal["clarify"]
    updated_constraints: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)

class VacationOption(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Allow extra fields from LLM
    option_id: str = ""
    destination: str
    location: str = ""
    est_total_usd: float = 0.0
    arrival_date: str = ""
    leave_date: str = ""
    highlights: List[str] = Field(default_factory=list)
    why_fits: str = ""
    offer_refs: Dict[str, str] = Field(default_factory=dict)

class RecommendationsEnvelope(BaseEnvelope):
    type: Literal["recommendations"]
    updated_constraints: Dict[str, Any] = Field(default_factory=dict)
    as_of_ts: int = 0
    currency: str = "USD"
    options: List[VacationOption] = Field(default_factory=list)

class BookingCopyEnvelope(BaseEnvelope):
    type: Literal["booking_copy"]
    title: str
    description: str

Envelope = Union[ClarifyEnvelope, RecommendationsEnvelope, BookingCopyEnvelope]
