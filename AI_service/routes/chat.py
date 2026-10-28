from fastapi import APIRouter, Depends

from AI_service.controllers.chat_controller import handle_chat, handle_send_message
from AI_service.dependencies import authenticated_user
from AI_service.schemas.chat import ChatRequest, ChatResponse, MessageRequest


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: dict = Depends(authenticated_user)
):
    return handle_chat(request, user)


@router.post(
    "/api/conversations/{conversation_id}/messages",
    response_model=ChatResponse
)
def send_chat_message(
    conversation_id: str,
    request: MessageRequest,
    user: dict = Depends(authenticated_user)
):
    return handle_send_message(conversation_id, request, user)
