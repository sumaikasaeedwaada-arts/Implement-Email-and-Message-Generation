import logging

from fastapi import HTTPException

from AI_service.errors import ChatUnavailableError, InvalidChatMessageError
from AI_service.schemas.chat import ChatRequest, MessageRequest
from AI_service.services.chat_service import (
    create_conversation_for_chat,
    send_message,
)

logger = logging.getLogger(__name__)


def handle_send_message(
    conversation_id: str,
    request: MessageRequest,
    user: dict
) -> dict:
    try:
        return send_message(conversation_id, request, user)
    except HTTPException:
        raise
    except InvalidChatMessageError as exc:
        raise HTTPException(status_code=422, detail=exc.public_message) from exc
    except ChatUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected chat processing failure.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        ) from exc


def handle_chat(request: ChatRequest, user: dict) -> dict:
    try:
        conversation_id = request.conversation_id
        if conversation_id is None:
            conversation_id = create_conversation_for_chat(user["id"])
        return handle_send_message(conversation_id, request, user)
    except HTTPException:
        raise
    except ChatUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from exc
    except Exception as exc:
        logger.exception("Unexpected chat request failure.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        ) from exc
