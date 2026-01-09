from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, Optional, Type

from pydantic import ValidationError

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TerminationCondition
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, ModelClientStreamingChunkEvent, TextMessage, StopMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils.config import settings
from agent.prompts import (
    CONVERSATION_SYSTEM,
    SEARCH_SYSTEM,
    BOOKING_SYSTEM,
)
from agent.schemas import (
    Envelope,
    ClarifyEnvelope,
    RecommendationsEnvelope,
    BookingCopyEnvelope,
)
from tools.tools import search_vacation_options

JSON_START = "JSON_START"
JSON_END = "JSON_END"


class ConversationTermination(TerminationCondition):
    """Terminate when ConversationAgent outputs JSON_END (final response to user)."""
    
    def __init__(self) -> None:
        self._terminated = False
    
    @property
    def terminated(self) -> bool:
        return self._terminated
    
    async def __call__(self, messages: list[AgentEvent | ChatMessage]) -> StopMessage | None:
        if not messages:
            return None
        last = messages[-1]
        # Terminate when ConversationAgent outputs JSON_END
        if isinstance(last, TextMessage) and last.source == "ConversationAgent" and JSON_END in last.content:
            self._terminated = True
            return StopMessage(content="ConversationAgent completed response", source="termination")
        return None
    
    async def reset(self) -> None:
        self._terminated = False


class AgentOutputParseError(Exception):
    pass


class VacationAgentService:
    """Router + Specialists vacation planner service.
    
    Architecture:
    - SelectorGroupChat routes to the right specialist based on context
    - ConversationAgent: Clarifies requirements, presents search results
    - SearchAgent: Calls vacation search tool
    - BookingAgent: Creates calendar copy (separate, post-confirmation)
    
    Benefits over RoundRobin:
    - 1-2 LLM calls per turn instead of 3
    - Dynamic routing based on conversation state
    - No wasted calls when info is incomplete
    """

    def __init__(self, logger, model_client: Optional[OpenAIChatCompletionClient] = None) -> None:
        self._logger = logger

        if model_client is not None:
            self._model_client = model_client
        else:
            if not settings.OPENAI_API_KEY.strip():
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._model_client = OpenAIChatCompletionClient(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )

        self._team = self._build_team()
        self._booking_agent = self._build_booking_agent()

    # ---------- parsing ----------
    def extract_json_envelope(self, text: str) -> Dict[str, Any]:
        if JSON_START not in text or JSON_END not in text:
            raise AgentOutputParseError("Missing JSON_START/JSON_END markers")

        m = re.search(
            rf"{re.escape(JSON_START)}\s*(\{{.*\}})\s*{re.escape(JSON_END)}",
            text,
            re.DOTALL,
        )
        if not m:
            raise AgentOutputParseError("Could not locate JSON envelope")
        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise AgentOutputParseError(f"Invalid JSON envelope: {e}") from e

    def validate_envelope(self, data: Dict[str, Any]) -> Envelope:
        typ = data.get("type")
        model_map: Dict[str, Type[Envelope]] = {
            "clarify": ClarifyEnvelope,
            "recommendations": RecommendationsEnvelope,
            "booking_copy": BookingCopyEnvelope,
        }
        if typ not in model_map:
            raise AgentOutputParseError(f"Unknown envelope type: {typ!r}")
        try:
            return model_map[typ].model_validate(data)  # type: ignore[return-value]
        except ValidationError as e:
            raise AgentOutputParseError(f"Envelope validation failed: {e}") from e

    # ---------- build agents ----------
    def _build_team(self) -> SelectorGroupChat:
        """Build Router + Specialists team using SelectorGroupChat."""
        
        async def tool_search_vacation_options(constraints: Dict[str, Any], session_id: str) -> Dict[str, Any]:
            return await search_vacation_options(constraints=constraints, session_id=session_id)

        # Conversation specialist - handles clarification and presenting results
        conversation_agent = AssistantAgent(
            name="ConversationAgent",
            description="Clarifies user requirements and presents search results. Call when gathering info or showing options.",
            model_client=self._model_client,
            system_message=CONVERSATION_SYSTEM,
            model_client_stream=True,
        )

        # Search specialist - calls the vacation search tool
        search_agent = AssistantAgent(
            name="SearchAgent",
            description="Searches for vacation options. Call ONLY when we have: destination, dates, budget, travelers.",
            model_client=self._model_client,
            system_message=SEARCH_SYSTEM,
            tools=[tool_search_vacation_options],
            model_client_stream=True,
        )

        # Selector prompt guides routing decisions
        selector_prompt = """You are the Router. Decide which agent should handle the next step.

AGENTS:
- ConversationAgent: Clarify requirements OR present search results to user
- SearchAgent: Search for vacations (only if we have destination + dates + budget + travelers)

ROUTING RULES:
1. Missing required info (destination/dates/budget/travelers)? → ConversationAgent
2. Have all required info and NO search results yet? → SearchAgent
3. SearchAgent just returned results? → ConversationAgent (to present them)
4. User asking questions or chatting? → ConversationAgent

Based on the conversation, select the next agent."""

        # Terminate when ConversationAgent outputs final JSON response
        termination = ConversationTermination() | MaxMessageTermination(10)
        
        return SelectorGroupChat(
            participants=[conversation_agent, search_agent],
            model_client=self._model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
        )

    def _build_booking_agent(self) -> AssistantAgent:
        return AssistantAgent(
            name="BookingAgent",
            description="Prepares calendar title/description text from selected option and booking links.",
            model_client=self._model_client,
            system_message=BOOKING_SYSTEM,
            model_client_stream=True,
        )

    # ---------- streaming runners ----------
    async def stream_plan_turn(
        self,
        *,
        session_id: str,
        session_constraints: Dict[str, Any],
        user_text: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Routes to appropriate specialist and streams SSE-friendly events.
        
        ConversationAgent is expected to emit the JSON envelope (clarify or recommendations).
        """
        task = (
            f"Session constraints so far: {json.dumps(session_constraints, ensure_ascii=False)}\n"
            f"User message: {user_text}\n"
            f"Session id: {session_id}\n"
            "Route to the appropriate specialist. ConversationAgent must output JSON_START/JSON_END envelope."
        )

        async for item in self._team.run_stream(task=task):
            if isinstance(item, ModelClientStreamingChunkEvent):
                yield {"event": "delta", "text": item.content}
                continue

            if isinstance(item, TextMessage) and item.source != "user":
                # Emit message per agent
                yield {"event": "message", "agent": item.source, "text": item.content}

                # Only ConversationAgent emits the final envelope
                if item.source != "ConversationAgent":
                    continue

                try:
                    data = self.extract_json_envelope(item.content)
                    env = self.validate_envelope(data)
                    yield {"event": "structured", "agent": "ConversationAgent", "data": env.model_dump()}
                except AgentOutputParseError as e:
                    self._logger.error("ConversationAgent envelope parse/validation error: %s", e)
                    yield {"event": "structured_error", "agent": "ConversationAgent", "message": str(e)}

    async def stream_booking_copy(
        self,
        *,
        selected_option: Dict[str, Any],
        booking_links: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs BookingAgent to produce booking_copy envelope."""
        task = (
            "Create booking copy for a calendar event.\n"
            f"Selected option JSON: {json.dumps(selected_option, ensure_ascii=False)}\n"
            f"Booking links JSON: {json.dumps(booking_links, ensure_ascii=False)}\n"
            "Output only JSON_START..JSON_END with type=booking_copy."
        )

        async for item in self._booking_agent.run_stream(task=task):
            if isinstance(item, ModelClientStreamingChunkEvent):
                yield {"event": "delta", "text": item.content}
                continue

            if isinstance(item, TextMessage) and item.source == "assistant":
                yield {"event": "message", "agent": "BookingAgent", "text": item.content}
                try:
                    data = self.extract_json_envelope(item.content)
                    env = self.validate_envelope(data)
                    if not isinstance(env, BookingCopyEnvelope):
                        raise AgentOutputParseError("BookingAgent must output type=booking_copy")
                    yield {"event": "structured", "agent": "BookingAgent", "data": env.model_dump()}
                except AgentOutputParseError as e:
                    self._logger.error("BookingAgent envelope parse/validation error: %s", e)
                    yield {"event": "structured_error", "agent": "BookingAgent", "message": str(e)}
