from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils.config import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance, PointStruct
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore

@dataclass
class PreferenceMemory:
    user_id: str
    text: str
    metadata: Dict[str, Any]
    created_at: float

class NoopLongTermMemory:
    async def add_preference(self, user_id: str, text: str, metadata: Dict[str, Any]) -> None:
        return

    async def search_preferences(self, user_id: str, query: str, top_k: int = 5) -> List[PreferenceMemory]:
        return []

class QdrantLongTermMemory:
    def __init__(self) -> None:
        if not settings.QDRANT_URL.strip():
            raise RuntimeError("QDRANT_URL not set")
        if QdrantClient is None:
            raise RuntimeError("qdrant-client not installed")
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection = settings.QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int) -> None:
        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection not in cols:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def add_preference(self, user_id: str, text: str, metadata: Dict[str, Any]) -> None:
        # Embeddings are created by the tool layer; this class just stores payload + vector.
        # This method is a placeholder to show structure.
        return

    async def search_preferences(self, user_id: str, query: str, top_k: int = 5) -> List[PreferenceMemory]:
        # Placeholder: requires embeddings integration.
        return []

def get_long_term_memory():
    if settings.QDRANT_URL.strip():
        return QdrantLongTermMemory()
    return NoopLongTermMemory()
