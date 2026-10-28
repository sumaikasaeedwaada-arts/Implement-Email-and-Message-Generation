from fastapi import Header, HTTPException

from AI_service import database, security


def authenticated_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    user_id = security.read_access_token(authorization[7:].strip())
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    with database.connect() as connection:
        user = database.row_dict(
            connection.execute(
                "SELECT id, email, created_at FROM users WHERE id = ?",
                (user_id,)
            ).fetchone()
        )
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return user


def owned_conversation(connection, conversation_id: str, user_id: int):
    conversation = connection.execute(
        "SELECT id, user_id, title, created_at, updated_at "
        "FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id)
    ).fetchone()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation
