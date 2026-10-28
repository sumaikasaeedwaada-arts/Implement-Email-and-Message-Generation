import json
import logging
import uuid

from fastapi import HTTPException

from AI_service import (
    ai_service,
    chat_api,
    customer_context,
    database,
    rag,
    settings,
)
from AI_service.dependencies import owned_conversation
from AI_service.errors import ChatUnavailableError, InvalidChatMessageError
from AI_service.schemas.chat import (
    MessageRequest,
    build_chat_prompt,
)
from AI_service.time_utils import utc_now


logger = logging.getLogger(__name__)


def create_conversation_for_chat(user_id: int) -> str:
    conversation_id = str(uuid.uuid4())
    now = utc_now()
    try:
        with database.connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, 'New conversation', ?, ?)",
                (conversation_id, user_id, now, now)
            )
    except Exception as exc:
        logger.exception("Chat conversation could not be created.")
        raise ChatUnavailableError(
            "The chat service is temporarily unavailable."
        ) from exc
    return conversation_id


def send_message(
    conversation_id: str,
    request: MessageRequest,
    user: dict
) -> dict:
    message = request.message.strip()
    if not message:
        raise InvalidChatMessageError("Message cannot be blank.")

    try:
        with database.connect() as connection:
            conversation = owned_conversation(
                connection,
                conversation_id,
                user["id"]
            )
            prior_messages = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (conversation_id, settings.MAX_HISTORY_MESSAGES)
            ).fetchall()
            history = [
                {"role": row["role"], "content": row["content"]}
                for row in reversed(prior_messages)
            ]
            retrieved = rag.search_chunks(connection, user["id"], message)
            identifier = (
                request.customer_identifier.strip()
                if request.customer_identifier
                else None
            )
            customer_matches = customer_context.retrieve_customer_context(
                connection,
                user["id"],
                identifier
            )
            try:
                retrieved_business = chat_api.search_knowledge(message)
            except Exception:
                logger.warning(
                    "Prepared business context retrieval failed; "
                    "continuing without business sources"
                )
                retrieved_business = []
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat context retrieval failed.")
        raise ChatUnavailableError(
            "Document search is temporarily unavailable."
        ) from exc

    retrieved_chunks = []
    seen_chunks = set()
    for item in customer_matches + retrieved:
        chunk_id = item.get("chunk_id")
        key = (
            ("id", chunk_id)
            if chunk_id is not None
            else (
                "content",
                item.get("document_id"),
                item.get("page_number"),
                item.get("content")
            )
        )
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        retrieved_chunks.append(item)
        if len(retrieved_chunks) >= settings.MAX_RETRIEVED_CHUNKS:
            break
    retrieved = retrieved_chunks

    citations = []
    context_chunks = []
    customer_references = []
    customer_chunk_ids = {match["chunk_id"] for match in customer_matches}
    for index, item in enumerate(retrieved, start=1):
        citations.append({
            "index": index,
            "document_id": item["document_id"],
            "filename": item["filename"],
            "page_number": item["page_number"],
            "score": round(item["score"], 4) if "score" in item else None
        })
        context_chunks.append({
            "citation": index,
            "document": item["filename"],
            "page": item["page_number"],
            "text": item["content"]
        })
        if item.get("chunk_id") in customer_chunk_ids:
            customer_references.append({
                "customer_identifier": identifier,
                "citation": index,
                "document": item["filename"],
                "page": item["page_number"]
            })

    prompt = build_chat_prompt(
        conversation["title"],
        history,
        retrieved_business,
        context_chunks,
        message,
        request.business_context,
        request.task_context,
        identifier,
        customer_references
    )

    try:
        answer = ai_service.generate_answer(prompt).strip()
        if not answer:
            raise ValueError("The model returned an empty response.")
    except Exception as exc:
        logger.error("Chat generation failed.")
        raise ChatUnavailableError(
            "The AI service is temporarily unavailable."
        ) from exc

    now = utc_now()
    title = conversation["title"]
    try:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owned_conversation(connection, conversation_id, user["id"])
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, citations_json, created_at) "
                "VALUES (?, 'user', ?, '[]', ?)",
                (conversation_id, message, now)
            )
            connection.execute(
                "INSERT INTO messages(conversation_id, role, content, citations_json, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (conversation_id, answer, json.dumps(citations), now)
            )
            if title == "New conversation":
                title = message[:70]
            connection.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id)
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat response could not be saved.")
        raise ChatUnavailableError(
            "The chat response could not be saved."
        ) from exc

    return {
        "conversation_id": conversation_id,
        "response": answer,
        "citations": citations,
        "created_at": now
    }
