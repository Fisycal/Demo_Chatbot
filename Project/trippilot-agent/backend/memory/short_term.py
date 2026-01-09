from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

from utils.config import settings

class SessionDoc(BaseModel):
    session_id: str
    created_at: float = Field(default_factory=lambda: time.time())
    status: str = "active"  # active|awaiting_user|awaiting_confirm|completed|error
    # conversational state:
    constraints: Dict[str, Any] = Field(default_factory=dict)
    last_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    selected_option: Optional[Dict[str, Any]] = None
    booking_links: Optional[Dict[str, Any]] = None
    calendar_event: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    clarify_questions: List[str] = Field(default_factory=list)
    clarify_missing_fields: List[str] = Field(default_factory=list)

class InMemoryShortTermStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionDoc] = {}
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        session_id = str(uuid4())
        async with self._lock:
            self._sessions[session_id] = SessionDoc(session_id=session_id)
            self._messages[session_id] = []
        return session_id

    async def get_session(self, session_id: str) -> SessionDoc:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError("session not found")
            return self._sessions[session_id]

    async def save_session(self, doc: SessionDoc) -> None:
        async with self._lock:
            self._sessions[doc.session_id] = doc

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        async with self._lock:
            self._messages[session_id].append({"ts": time.time(), "role": role, "content": content})

    async def list_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            msgs = self._messages.get(session_id, [])
            return msgs[-limit:]

class MongoShortTermStore:
    def __init__(self) -> None:
        self._client = AsyncIOMotorClient(settings.MONGO_URI, uuidRepresentation="standard")
        self._db = self._client[settings.MONGO_DB]
        self._sessions = self._db["sessions"]
        self._messages = self._db["messages"]

    async def create_session(self) -> str:
        session_id = str(uuid4())
        doc = SessionDoc(session_id=session_id).model_dump()
        await self._sessions.insert_one(doc)
        return session_id

    async def get_session(self, session_id: str) -> SessionDoc:
        doc = await self._sessions.find_one({"session_id": session_id}, {"_id": 0})
        if not doc:
            raise KeyError("session not found")
        return SessionDoc(**doc)

    async def save_session(self, doc: SessionDoc) -> None:
        await self._sessions.update_one(
            {"session_id": doc.session_id},
            {"$set": doc.model_dump()},
            upsert=True,
        )

    async def append_message(self, session_id: str, role: str, content: str) -> None:
        await self._messages.insert_one({
            "session_id": session_id,
            "ts": time.time(),
            "role": role,
            "content": content,
        })

    async def list_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self._messages.find({"session_id": session_id}, {"_id": 0}).sort("ts", 1)
        docs = await cursor.to_list(length=limit)
        return docs[-limit:]

def get_short_term_store():
    if settings.MONGO_URI.strip():
        return MongoShortTermStore()
    return InMemoryShortTermStore()

# --- SSE event bus (single-process) ---
# For production multi-worker deployments, replace this with Redis Streams / PubSub.
class EventBus:
    def __init__(self) -> None:
        self._queues: Dict[str, "asyncio.Queue[dict]"] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def publish(self, session_id: str, event: dict) -> None:
        async with self._lock:
            q = self._queues.setdefault(session_id, asyncio.Queue(maxsize=1000))
        # best-effort: if queue is full, drop oldest
        if q.full():
            try:
                _ = q.get_nowait()
                self._logger.warning("event queue full for session %s; dropping oldest event", session_id)
            except Exception:
                pass
        await q.put(event)

    async def subscribe(self, session_id: str) -> AsyncGenerator[dict, None]:
        async with self._lock:
            q = self._queues.setdefault(session_id, asyncio.Queue(maxsize=1000))
        while True:
            event = await q.get()
            yield event
