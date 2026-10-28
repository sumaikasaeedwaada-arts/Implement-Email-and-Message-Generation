import hmac
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from google.genai import types
from pydantic import BaseModel, Field

from AI_service import ai_service, settings
from AI_service.context.business_context import KNOWLEDGE_BASE
from AI_service.ai_service import MODEL, SYSTEM_PROMPT

app = FastAPI()
logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 20
MAX_CONVERSATIONS = 1000
MIN_RELEVANCE_SCORE = settings.BUSINESS_KNOWLEDGE_MIN_RELEVANCE_SCORE
client = ai_service.client

conversations = OrderedDict()
knowledge_base = KNOWLEDGE_BASE
vector_store = []
_vector_store_signature = None


def create_embedding(text, task_type):
    return ai_service.create_embedding(text, task_type)


def validate_embedding(embedding):
    if not isinstance(embedding, (list, tuple)) or not embedding:
        raise ValueError("Embedding must be a non-empty numeric vector.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in embedding
    ):
        raise ValueError("Embedding must contain only numeric values.")
    try:
        vector = [float(value) for value in embedding]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Embedding must contain only numeric values.") from exc
    if not all(np.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain only finite numeric values.")
    if not any(vector):
        raise ValueError("Embedding must not be a zero vector.")
    return vector


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size or not a.size:
        return 0.0
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _public_knowledge_signature():
    return tuple(
        (
            item.get("id"),
            item.get("title"),
            item.get("content"),
            item.get("source"),
            item.get("is_public")
        )
        for item in knowledge_base
        if isinstance(item, dict) and item.get("is_public") is True
    )


def build_vector_store():
    global _vector_store_signature
    indexed_items = []
    seen_ids = set()
    dimensions = None
    for item in knowledge_base:
        if not isinstance(item, dict) or item.get("is_public") is not True:
            continue
        item_id = item.get("id")
        title = item.get("title")
        content = item.get("content")
        source = item.get("source")
        if (
            isinstance(item_id, bool)
            or not isinstance(item_id, (str, int))
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(content, str)
            or not content.strip()
            or not isinstance(source, str)
        ):
            logger.warning("Skipping malformed public knowledge entry.")
            continue
        if item_id in seen_ids:
            logger.warning(
                "Skipping duplicate public knowledge entry (id=%s)",
                item_id
            )
            continue
        embedding = validate_embedding(
            create_embedding(content.strip(), "RETRIEVAL_DOCUMENT")
        )
        if dimensions is None:
            dimensions = len(embedding)
        elif len(embedding) != dimensions:
            raise ValueError(
                "Business knowledge embeddings have inconsistent dimensions."
            )
        indexed_items.append({
            "id": item_id,
            "title": title.strip(),
            "content": content.strip(),
            "source": source.strip(),
            "is_public": True,
            "metadata": {
                "id": item_id,
                "title": title.strip(),
                "source": source.strip(),
                "is_public": True
            },
            "embedding": embedding
        })
        seen_ids.add(item_id)
    vector_store[:] = indexed_items
    _vector_store_signature = _public_knowledge_signature()
    return vector_store


def search_knowledge(
    question,
    limit=None,
    filters=None
):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Search query must not be empty.")
    if limit is None:
        limit = settings.BUSINESS_KNOWLEDGE_TOP_K
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Search limit must be a positive integer.")
    if filters is None:
        filters = {}
    if not isinstance(filters, dict):
        raise ValueError("Search filters must be a mapping.")
    filter_fields = {"id", "title", "source"}
    unsupported_filters = set(filters) - filter_fields
    if unsupported_filters:
        raise ValueError(
            "Unsupported business knowledge filter: "
            + ", ".join(sorted(str(key) for key in unsupported_filters))
        )

    signature = _public_knowledge_signature()
    if (
        not vector_store
        or _vector_store_signature != signature
    ):
        build_vector_store()
    if not vector_store:
        return []

    query_embedding = validate_embedding(
        create_embedding(question.strip(), "RETRIEVAL_QUERY")
    )
    if len(query_embedding) != len(vector_store[0]["embedding"]):
        raise ValueError(
            "Query embedding dimensions do not match business knowledge."
        )
    results = []
    for item in vector_store:
        if item.get("is_public") is not True:
            continue
        if any(item.get(key) != value for key, value in filters.items()):
            continue
        score = cosine_similarity(query_embedding, item["embedding"])
        if score >= MIN_RELEVANCE_SCORE:
            results.append({
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "source": item["source"],
                "metadata": dict(item["metadata"]),
                "score": score
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    user: str = Field(max_length=1000)
    business: str = Field(max_length=2000)
    customer: str = Field(max_length=2000)
    task: str = Field(max_length=2000)
    notes: str = Field(max_length=4000)


class KnowledgeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)
    is_public: bool = False


@app.get("/")
def home():
    return {"message": "XICTEK Systems and HisabDo AI Chat API is running"}


@app.post("/knowledge")
def add_knowledge(
    request: KnowledgeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_knowledge_api_key(x_api_key)

    new_item = {
        "id": len(knowledge_base) + 1,
        "title": request.title,
        "content": request.content,
        "source": "Admin-provided knowledge",
        "is_public": request.is_public
    }
    knowledge_base.append(new_item)
    vector_store.clear()
    global _vector_store_signature
    _vector_store_signature = None
    return {
        "message": "Knowledge added successfully",
        "knowledge": new_item
    }


@app.get("/knowledge")
def get_knowledge(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_knowledge_api_key(x_api_key)
    return {"knowledge_base": knowledge_base}


def require_knowledge_api_key(provided_api_key: str | None) -> None:
    expected_api_key = os.getenv("XICTEK_KNOWLEDGE_API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="Knowledge updates are not configured."
        )
    if not provided_api_key or not hmac.compare_digest(
        provided_api_key,
        expected_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.post("/chat")
def chat(request: ChatRequest):
    history = conversations.get(request.conversation_id, [])
    prompt_data = {
        "business_context": {
            "user": request.user,
            "business": request.business,
            "customer": request.customer,
            "task": request.task,
            "notes": request.notes
        },
        "conversation_history": history[-10:],
        "current_question": request.message
    }

    try:
        prompt_data["retrieved_knowledge"] = search_knowledge(request.message)
        response = client.models.generate_content(
            model=MODEL,
            contents=json.dumps(prompt_data, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
    except Exception as exc:
        logger.error("Chat generation failed.")
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again later."
            )
        ) from exc

    timestamp = datetime.now().isoformat()
    history.extend([
        {"role": "user", "message": request.message, "time": timestamp},
        {"role": "assistant", "message": response.text, "time": timestamp}
    ])
    conversations[request.conversation_id] = history[-MAX_HISTORY_MESSAGES:]
    conversations.move_to_end(request.conversation_id)
    while len(conversations) > MAX_CONVERSATIONS:
        conversations.popitem(last=False)

    return {
        "conversation_id": request.conversation_id,
        "question": request.message,
        "retrieved_knowledge": prompt_data["retrieved_knowledge"],
        "response": response.text
    }


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "messages": conversations.get(conversation_id, [])
    }
