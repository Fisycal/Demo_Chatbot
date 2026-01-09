import os
import uuid
import re
import hashlib
import logging
import datetime
import time
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import httpx
import torch
from langchain_core.documents import Document
from passlib.context import CryptContext
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from openai import AsyncOpenAI
import numpy as np
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue

# -----------------------------------------------------------------------------
# SYSTEM PROMPTS
# -----------------------------------------------------------------------------
def system_prompt_general_info():
    """System prompt for document Q&A chatbot"""
    return """You are a helpful document Q&A assistant. Your role is to answer questions based ONLY on the provided document context.

Instructions:
1. Answer questions using ONLY the information from the context snippets provided
2. If the context contains the answer, provide a clear, accurate response
3. Quote relevant parts from the context when appropriate to support your answer
4. If the context doesn't contain enough information to answer the question, clearly state: "I cannot find this information in the provided documents."
5. Be concise but complete - include all relevant details from the context
6. Maintain a professional and helpful tone
7. If asked about page numbers or sources, refer to the metadata provided

Remember: Never make up information or use knowledge outside the provided context."""
# -----------------------------------------------------------------------------
# ENV & LOGGER
# -----------------------------------------------------------------------------
load_dotenv()
# load_dotenv("env.server")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("async_qdrant_indexer")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DOC_SOURCE_PATH      = "knowledge"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "My-DOCU")
OLLAMA_ENDPOINT      = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_API_BASE      = OLLAMA_ENDPOINT.rstrip("/") + "/v1"
QDRANT_ENDPOINT      = os.getenv("QDRANT_ENDPOINT", "http://localhost:6333")
OPEN_AI_API_KEY      = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Optimized chunking for general documents (PDFs, DOCX)
CHUNK_SIZE           = 1000
CHUNK_OVERLAP        = 150

BATCH_SIZE           = 1500
EMBED_BATCH_SIZE     = 50
EMBED_SEMAPHORE_SIZE = 10
EMBEDDING_DIM        = 1024
SIMILARITY_THRESHOLD = 0.8
TOP_K_CITATIONS      = 5
# Important: do NOT reset collections by default; make it opt-in via env
RESET_COLLECTION     = os.getenv("RESET_COLLECTION","false").lower() in {"1","true","yes"}
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Number of context snippets to pass to LLM for Q&A
INFO_TOP_K = int(os.getenv("INFO_TOP_K", "5"))  # Increased for better context

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Embedding device: {device}")

# Thread pool & semaphore
thread_executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
embed_semaphore = asyncio.Semaphore(EMBED_SEMAPHORE_SIZE)

# -----------------------------------------------------------------------------
# QDRANT CLIENT
# -----------------------------------------------------------------------------
# Avoid using API key over insecure HTTP to prevent warnings and insecure usage.
# If the endpoint is HTTPS, include the API key; otherwise, omit it.
if QDRANT_ENDPOINT.lower().startswith("https"):
    q_client = QdrantClient(url=QDRANT_ENDPOINT, prefer_grpc=False, timeout=60, api_key=QDRANT_API_KEY)
else:
    q_client = QdrantClient(url=QDRANT_ENDPOINT, prefer_grpc=False, timeout=60)

def _collection_exists(name: str) -> bool:
    try:
        return q_client.collection_exists(name)
    except Exception:
        try:
            q_client.get_collection(name)
            return True
        except Exception:
            return False

def _ensure_collection(collection_name: str, reset: bool = False):
    """Generic collection creation helper"""
    if reset:
        if _collection_exists(collection_name):
            logger.info(f"Reset requested: deleting existing collection {collection_name}…")
            q_client.delete_collection(collection_name)
            time.sleep(1)
        logger.info(f"Creating collection {collection_name}…")
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            },
            timeout=120,
        )
        logger.info(f"Collection '{collection_name}' ready (reset mode).")
        return

    if _collection_exists(collection_name):
        logger.info(f"Collection {collection_name} exists; skipping create.")
        return
    logger.info(f"Collection {collection_name} missing; creating…")
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        },
        timeout=120,
    )
    logger.info(f"Collection '{collection_name}' ready.")
    
def ensure_collection_by_name(collection_name: str):
    """Ensure a collection exists by name"""
    if collection_name in ["My-DOCU"]:
        _ensure_collection(collection_name, RESET_COLLECTION)
    else:
        logger.warning(f"Unknown collection name: {collection_name}")

ensure_collection_by_name(COLLECTION_NAME)
# -----------------------------------------------------------------------------
# EMBEDDINGS VIA OLLAMA (/api/embed)
# -----------------------------------------------------------------------------
class OllamaEmbeddings(Embeddings):
    # def __init__(self, model: str = "nomic-embed-text", endpoint: str = OLLAMA_ENDPOINT):
    def __init__(self, model: str = "mxbai-embed-large:335m", endpoint: str = OLLAMA_ENDPOINT):
        self.model = model
        self.endpoint = endpoint

    async def _async_embed_documents(self, texts: List[str], batch_size: int) -> List[List[float]]:
        results: List[List[float]] = []
        async with httpx.AsyncClient(timeout=120) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                async with embed_semaphore:
                    resp = await client.post(
                        f"{self.endpoint}/api/embed",
                        json={"model": self.model, "input": batch},
                    )
                resp.raise_for_status()
                payload = resp.json()
                results.extend(payload["embeddings"])
        return results

    async def _async_embed_query(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.endpoint}/api/embed",
                json={"model": self.model, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"][0]

    def embed_documents(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
        return asyncio.get_event_loop().run_until_complete(
            self._async_embed_documents(texts, batch_size)
        )

    def embed_query(self, text: str) -> List[float]:
        return asyncio.get_event_loop().run_until_complete(self._async_embed_query(text))

# -----------------------------------------------------------------------------
# PROMPTS & UTILITIES
# -----------------------------------------------------------------------------

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def compute_pid(text: str) -> str:
    return str(uuid.UUID(compute_hash(text)[:32]))

def ensure_metadata(doc, source_file: str, page_number: int = 1, chapter: Optional[str] = None):
    meta = {"source": os.path.basename(source_file), "page": page_number}
    if chapter:
        meta["chapter"] = chapter
    doc.metadata["meta"] = meta
    return doc

def extract_metadata(doc) -> Dict[str, Any]:
    payload = doc.metadata.get("meta", {})
    return {"source": payload.get("source", "unknown"), "page": payload.get("page", "unknown")}

def serialize(h):
    out = []
    for m in h:
        mm = dict(m)
        if isinstance(mm.get("timestamp"), datetime.datetime):
            mm["timestamp"] = mm["timestamp"].isoformat()
        out.append(mm)
    return out

# -----------------------------------------------------------------------------
# Upload Directory
# -----------------------------------------------------------------------------
UPLOAD_DIR = "knowledge"

# -----------------------------------------------------------------------------
# INDEXING
# -----------------------------------------------------------------------------
def load_and_split(file_path: str, collection_name: str = None):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return []

    docs = loader.load()

    for doc in docs:
        page = doc.metadata.get("page") or doc.metadata.get("page_number") or 1
        ensure_metadata(doc, file_path, int(page))

    # Smart chunking for general documents - respects paragraphs, headings, and lists
    separators = [
        "\n\n",      # Paragraph breaks
        "\n### ",    # Markdown headings
        "\n## ",
        "\n# ",
        "\n- ",      # List items
        "\n• ",
        "\n* ",
        "\n",        # Line breaks
        ". ",        # Sentences
        " "          # Words (last resort)
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)

    try:
        existing = {p.id for p in q_client.scroll(collection_name, limit=10000)[0]}
    except Exception:
        existing = set()

    
    contents, metas, pids = [], [], []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue
        pid = compute_pid(text)
        if pid in existing:
            continue
        contents.append(text)
        metas.append(chunk.metadata["meta"])
        pids.append(pid)
    return pids, contents, metas

def ensure_collection_exists(collection_name: str):
    from qdrant_client.http.models import Distance, VectorParams
    if not q_client.collection_exists(collection_name=collection_name):
        logger.info(f"Creating Qdrant collection: {collection_name}")
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

async def update_qdrant_index_async(file_path: str, collection_name: str = None) -> int:
    use_collection = collection_name or COLLECTION_NAME
    
    pids, contents, metas = await asyncio.get_event_loop().run_in_executor(
        thread_executor, lambda: load_and_split(file_path, collection_name)
    )
    if not contents:
        return 0

    # Generate dense vectors (semantic)
    dense_vectors = await OllamaEmbeddings()._async_embed_documents(contents, EMBED_BATCH_SIZE)
    
    total = 0
    for i in range(0, len(contents), BATCH_SIZE):
        batch_points = [
                PointStruct(
                    id=pids[j],
                    vector={"dense": dense_vectors[j]},
                    payload={"text": contents[j], "meta": metas[j]}
                )
                for j in range(i, min(i + BATCH_SIZE, len(contents)))
            ]
        
        await asyncio.get_event_loop().run_in_executor(
            thread_executor,
            lambda pts=batch_points: q_client.upsert(collection_name=use_collection, points=pts),
        )
        total += len(batch_points)
    
    
    logger.info(f"Indexed {total} chunks with dense vectors from {file_path} into {use_collection}")
    return total


async def update_qdrant_index(file_path: str, collection_name: str = None) -> int:
    return await update_qdrant_index_async(file_path, collection_name=collection_name)

# -----------------------------------------------------------------------------
# VectorStore & Assistant wiring — reuse the same q_client
# -----------------------------------------------------------------------------
embedder = OllamaEmbeddings()

# -----------------------------------------------------------------------------
# Admin helpers: list & delete documents by filename in a collection
# -----------------------------------------------------------------------------
def _extract_filename_from_payload(payload: Dict[str, Any]) -> str:
    meta = (payload or {}).get("meta", {}) or {}
    # Try common keys across historical payloads
    return (
        meta.get("source")
        or payload.get("source")
        or meta.get("file")
        or payload.get("file")
        or payload.get("filename")
        or "unknown"
    )


def list_documents_in_collection(collection_name: str) -> List[Dict[str, Any]]:
    """Aggregate Qdrant points by filename across meta.source and legacy keys."""
    ensure_collection_exists(collection_name)
    filename_counts: Dict[str, int] = {}
    next_offset = None
    while True:
        points, next_offset = q_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=1000,
            offset=next_offset,
        )
        if not points:
            break
        for p in points:
            fname = _extract_filename_from_payload(p.payload or {})
            filename_counts[fname] = filename_counts.get(fname, 0) + 1
        if next_offset is None:
            break
    return [
        {"filename": fname, "point_count": count}
        for fname, count in sorted(filename_counts.items(), key=lambda x: x[0].lower())
    ]


def _count_points_for_filename(collection_name: str, filename: str) -> int:
    """Count points for a single filename by scrolling with OR filter (meta.source or source)."""
    total = 0
    next_offset = None
    flt = Filter(should=[
        FieldCondition(key="meta.source", match=MatchValue(value=filename)),
        FieldCondition(key="source", match=MatchValue(value=filename)),
    ])
    while True:
        points, next_offset = q_client.scroll(
            collection_name=collection_name,
            with_payload=False,
            limit=1000,
            offset=next_offset,
            scroll_filter=flt,
        )
        if not points:
            break
        total += len(points)
        if next_offset is None:
            break
    return total


def delete_documents_by_filenames(collection_name: str, filenames: List[str]) -> Dict[str, int]:
    """
    Delete all points whose payload.meta.source is in filenames.
    Returns a map filename -> estimated deleted count (counted pre-delete).
    """
    ensure_collection_exists(collection_name)
    filenames = [f for f in filenames if f]
    counts: Dict[str, int] = {}
    for f in filenames:
        try:
            counts[f] = _count_points_for_filename(collection_name, f)
        except Exception:
            counts[f] = 0
    if not filenames:
        return counts

    # Delete by both meta.source and top-level source to cover legacy data
    if filenames:
        flt_meta = Filter(must=[FieldCondition(key="meta.source", match=MatchAny(any=filenames))])
        q_client.delete(collection_name=collection_name, points_selector=flt_meta)
        flt_top = Filter(must=[FieldCondition(key="source", match=MatchAny(any=filenames))])
        q_client.delete(collection_name=collection_name, points_selector=flt_top)
    return counts

# -----------------------------------------------------------------------------
# MongoDB
# -----------------------------------------------------------------------------
mongo_conn = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
mongo_client = MongoClient(mongo_conn, serverSelectionTimeoutMS=3000)
conversation_collection = mongo_client["demochatbot"]["demo-conversations"]
users_collection = mongo_client["demochatbot"]["users"]
# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
try:
    mongo_client.server_info()
    logger.info("Connected to MongoDB.")
except Exception as e:
    logger.warning(f"MongoDB warning: {e}")


# -------------------------
# User management helpers
# -------------------------
def get_user_by_email(email: Optional[str]) -> Optional[Dict[str, Any]]:
    if not email:
        return None
    return users_collection.find_one({"email": email.lower()})


def create_user(email: str, password: str, is_admin: bool = False) -> Dict[str, Any]:
    """Create a new user with hashed password. Raises Exception if user exists."""
    if not email or not password:
        raise Exception("Email and password are required")
    existing = get_user_by_email(email)
    if existing:
        raise Exception("User already exists")
    hashed = pwd_context.hash(password)
    user = {
        "email": email.lower(),
        "password_hash": hashed,
        "is_admin": bool(is_admin),
        "created_at": datetime.datetime.utcnow(),
    }
    users_collection.insert_one(user)
    user.pop("password_hash", None)
    return user


def verify_user_credentials(email: str, password: str) -> bool:
    user = get_user_by_email(email)
    if not user:
        return False
    hashed = user.get("password_hash")
    if not hashed:
        return False
    try:
        return pwd_context.verify(password, hashed)
    except Exception:
        return False


def update_user_password(email: str, new_password: str) -> bool:
    """Update an existing user's password (hashes the new password).

    Returns True if the password was updated, False otherwise.
    """
    if not email or not new_password:
        raise Exception("Email and new password are required")
    user = get_user_by_email(email)
    if not user:
        raise Exception("User not found")
    new_hashed = pwd_context.hash(new_password)
    res = users_collection.update_one({"email": email.lower()}, {"$set": {"password_hash": new_hashed}})
    return res.modified_count > 0

# -----------------------------------------------------------------------------
# Streaming assistant with OpenAI-compatible Chat endpoint
# -----------------------------------------------------------------------------
class StreamingOllamaAssistant:
    def __init__(self):
        # self.model_name = os.getenv("OLLAMA_MODEL", " ")#can replace ollama with open ai
        # self.api_base = OLLAMA_API_BASE #canremove this for openai
        # self.headers = {"Authorization": "Bearer ollama"} #this can go too
        # self.client = AsyncOpenAI(base_url=self.api_base, api_key="ollama") #replace baseurl with openai key
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")#can replace ollama with open ai
        #self.api_base = OLLAMA_API_BASE #canremove this for openai
        #self.headers = {"Authorization": "Bearer ollama"} #this can go too
        self.client = AsyncOpenAI(api_key=OPEN_AI_API_KEY) #replace baseurl with openai key

    def get_history(self, person_id: str, session_id: str = None) -> List[Dict[str, Any]]:
        doc = conversation_collection.find_one({"personId": person_id})
        if not doc:
            conversation_collection.insert_one({"personId": person_id, "sessions": []})
            return []
        if session_id:
            pipeline = [
                {"$match": {"personId": person_id}},
                {"$project": {
                    "sessions": {
                        "$filter": {
                            "input": "$sessions",
                            "as": "s",
                            "cond": {"$eq": ["$$s.sessionId", session_id]},
                        }
                    }
                }},
            ]
            res = list(conversation_collection.aggregate(pipeline))
            if res and res[0].get("sessions"):
                return res[0]["sessions"][0].get("messages", [])
            return []
        msgs = []
        for s in doc.get("sessions", []):
            msgs.extend(s.get("messages", []))
        return msgs

    def save_history(self, person_id: str, session_id: str, history: List[Dict[str, Any]]):
        query = {"personId": person_id, "sessions.sessionId": session_id} 
        if conversation_collection.find_one(query):
            conversation_collection.update_one(
                query,
                {"$set": {
                    "sessions.$.messages": history,
                    "sessions.$.last_updated": datetime.datetime.utcnow(),
                }},
            )
        else:
            conversation_collection.update_one(
                {"personId": person_id},
                {"$push": {
                    "sessions": {
                        "sessionId": session_id,
                        "messages": history,
                        "created_at": datetime.datetime.utcnow(),
                        "last_updated": datetime.datetime.utcnow(),
                    }
                }},
                upsert=True,
            )

    async def process_info_message(
        self,
        person_id: str,
        user_input: str,
        document_context: Any = None,
        user_profile_context: Any = None,
        session_id: str = None,
        reload_system_prompt: bool = True,
        selected_model: str = None,
        collection_name: str = None,
    ):
        """Handle Info mode - LLM with context snippets"""
        results: List[Tuple[Any, float]] = []
        filtered: List[Tuple[Any, float]] = []
        previews: List[Tuple[str, int, str]] = []

        use_collection = collection_name or COLLECTION_NAME
        
        # Count documents in the collection
        try:
            doc_count = q_client.count(collection_name=use_collection, exact=True)
            if isinstance(doc_count, dict) and "count" in doc_count:
                doc_count = doc_count["count"]
        except Exception as e:
            logger.warning(f"Could not count documents in collection {use_collection}: {e}")
            doc_count = None
        
        try:
            #Vector search for general info
            q_emb = await embedder._async_embed_query(user_input)
            print(f'length of q_emb from search: {len(q_emb)}')

            response = q_client.query_points(
            collection_name="My-DOCU",
            query=q_emb,      # your embedding
            using="dense", # important since your collection uses "dense"
            limit=10,
            with_payload=True                 
        )
            
            for hit in response.points:
                payload = hit.payload or {}
                text = payload.get("text", "")
                meta = payload.get("meta", {})
                doc = type("Doc", (), {})()
                doc.page_content = text
                doc.metadata = {"meta": meta}
                results.append((doc, hit.score))
                print("Score:", hit.score)
                print("Payload:", hit.payload)
       

            results.sort(key=lambda x: x[1], reverse=True)
            # Use adaptive threshold: prefer high-quality matches but ensure we get enough context
            threshold = 0  # Lower threshold for better recall 0.75
            filtered = [(d, s) for d, s in results if s >= threshold]
            
            # Always try to get INFO_TOP_K results for better context
            if len(filtered) >= INFO_TOP_K:
                top_hits = filtered[:INFO_TOP_K]
            elif len(filtered) > 0:
                # Use filtered results + some lower-scored ones to reach INFO_TOP_K
                top_hits = filtered + results[len(filtered):INFO_TOP_K]
            else:
                # If nothing passes threshold, take top INFO_TOP_K anyway
                top_hits = results[:INFO_TOP_K]
            
            logger.info(f"Retrieved {len(top_hits)} chunks with scores: {[f'{s:.3f}' for _, s in top_hits[:3]]}")

            for d, _ in top_hits:
                chunk = d.page_content.strip()
                md = extract_metadata(d)
                previews.append((md["source"], md["page"], chunk))

        except Exception as e:
            logger.warning(f"Info retrieval error: {e}")

        # If no snippets were retrieved, avoid calling the LLM to prevent hallucination
        if not previews:
            fallback_msg = "I couldn't find any supporting information in the document."
            logger.info("Info: No retrieved chunks found, returning fallback.")
            # Prepare and save minimal history entry
            history = self.get_history(person_id, session_id) if not reload_system_prompt else []
            history.append({
                "role": "assistant",
                "content": fallback_msg,
                "timestamp": datetime.datetime.utcnow(),
                "model_used": "INFO_FALLBACK"
            })
            self.save_history(person_id, session_id, history)
            yield fallback_msg
            return

        # Prepare user input for history
        user_block = f"User Message:\n{user_input}"
        profile_block = f"User Profile Context: {user_profile_context}\n\n" if user_profile_context else ""
        full_user = f"{profile_block}{user_block}".strip()
        
        # Get existing history
        history = self.get_history(person_id, session_id) if not reload_system_prompt else []
        system_msg = {
            "role": "system",
            "content": system_prompt_general_info()
        }

        # Inject snippets for context with source information
        if previews:
            lines = []
            for i, (source, page, chunk) in enumerate(previews, 1):
                snippet = chunk.strip()
                lines.append(f"[Context {i} - Source: {source}, Page: {page}]\n{snippet}")
            snippets = "\n\n---\n\n".join(lines)
            
            context_message = f"""Here are the relevant excerpts from the documents to answer the user's question:{snippets}--- Use ONLY the information above to answer the user's question. If the answer is not in these excerpts, say so."""
            
            history = [
                system_msg,
                {"role": "user", "content": context_message}
            ] + history
        else:
            history = [system_msg] + history

        history.append({"role": "user", "content": full_user, "timestamp": datetime.datetime.utcnow()})

        # Call LLM
        def serialize(h):
            out = []
            for m in h:
                mm = dict(m)
                if isinstance(mm.get("timestamp"), datetime.datetime):
                    mm["timestamp"] = mm["timestamp"].isoformat()
                out.append(mm)
            return out

        formatted = serialize(history)
        model_to_use = selected_model or self.model_name
        logger.info(f"🔁 Info mode calling model: {model_to_use}")

        full_resp = ""
        try:
            stream_resp = await self.client.chat.completions.create(
                model=model_to_use,
                messages=formatted,
                temperature=0.0,
                stream=True,
                stream_options={"include_usage": True}
            )
            async for event in stream_resp:
                if event.choices:
                    delta = event.choices[0].delta.content
                    if delta is not None:              # ✅ check before using
                        full_resp += delta
                        yield delta
                    else:
                        # Log skipped events for visibility
                        logger.debug(f"Skipped empty delta event: {event.choices[0].delta}")

            # Show sources for info mode
            if previews:
                numbered = [
                    f"{i}. **{src}** (page {pg}):\n{chunk}"
                    for i, (src, pg, chunk) in enumerate(previews, start=1)
                ]
                yield "\n\n**Top Sources & Chunks:**\n" + "\n\n".join(numbered)

            # Save history
            history.append({
                "role": "assistant",
                "content": full_resp,
                "timestamp": datetime.datetime.utcnow(),
                "model_used": model_to_use
            })
            self.save_history(person_id, session_id, history)

        except Exception as e:
            logger.error(f"Info mode LLM error: {e}")
            yield f"\n[Error generating response: {e}]"
