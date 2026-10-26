import logging
import os
import sqlite3
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from starlette.middleware.base import BaseHTTPMiddleware

from AI_service import (
    ai_service,
    chat_api,
    customer_context,
    database,
    rag,
    security,
    settings
)
from AI_service.dependencies import authenticated_user, owned_conversation
from AI_service.routes.chat import router as chat_router
from AI_service.schemas.business_summary import (
    BusinessSummaryRequest,
    BusinessSummaryResponse,
)
from AI_service.schemas.chat import (
    BusinessContext,
    MessageRequest,
    TaskContext,
    build_chat_prompt,
    context_fields,
)
from AI_service.services.business_summary_service import (
    BusinessSummaryUnavailableError,
    InvalidBusinessSummaryError,
    generate_business_summary,
)
from AI_service.time_utils import utc_now


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)
STATIC_DIRECTORY = os.path.join(os.path.dirname(__file__), "static")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class ConversationRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=160)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    database.initialize_database()
    if not (os.getenv("AUTH_SECRET") or settings.AUTH_SECRET):
        logger.warning(
            "AUTH_SECRET is unset; access tokens are process-local. "
            "Configure a stable secret before deployment."
        )
    yield


class SQLiteRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            path.startswith("/api/") or path == "/chat"
        ) and path != "/api/health":
            client_host = request.client.host if request.client else "unknown"
            category = "auth" if path in {
                "/api/auth/login",
                "/api/auth/register"
            } else "api"
            limit = 10 if category == "auth" else 120
            bucket_key = f"{category}:{client_host}"
            now = time.time()
            with database.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                bucket = connection.execute(
                    "SELECT window_started, request_count FROM rate_limit_buckets "
                    "WHERE bucket_key = ?",
                    (bucket_key,)
                ).fetchone()
                if bucket is None or now - bucket["window_started"] >= 60:
                    connection.execute(
                        """
                        INSERT INTO rate_limit_buckets
                            (bucket_key, window_started, request_count)
                        VALUES (?, ?, 1)
                        ON CONFLICT(bucket_key) DO UPDATE SET
                            window_started = excluded.window_started,
                            request_count = 1
                        """,
                        (bucket_key, now)
                    )
                    count = 1
                else:
                    count = bucket["request_count"] + 1
                    if count <= limit:
                        connection.execute(
                            "UPDATE rate_limit_buckets SET request_count = ? "
                            "WHERE bucket_key = ?",
                            (count, bucket_key)
                        )
                if count > limit:
                    retry_after = max(
                        1,
                        int(60 - (now - bucket["window_started"]))
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded. Please try again shortly."
                        },
                        headers={"Retry-After": str(retry_after)}
                    )
        return await call_next(request)


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Authenticated business chat with document-backed RAG.",
    lifespan=lifespan
)
app.add_middleware(SQLiteRateLimitMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIRECTORY, "index.html"))


@app.get("/api/health")
def health():
    with database.connect() as connection:
        connection.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/api/auth/register", status_code=201)
def register(request: RegisterRequest):
    email = str(request.email).strip().lower()
    now = utc_now()
    with database.connect() as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users(email, password_hash, created_at) "
                "VALUES (?, ?, ?)",
                (email, security.hash_password(request.password), now)
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="An account with this email already exists."
            ) from exc
    user_id = cursor.lastrowid
    return {
        "access_token": security.create_access_token(user_id),
        "token_type": "bearer",
        "user": {"id": user_id, "email": email, "created_at": now}
    }


@app.post("/api/auth/login")
def login(request: LoginRequest):
    with database.connect() as connection:
        user = connection.execute(
            "SELECT id, email, password_hash, created_at FROM users "
            "WHERE email = ? COLLATE NOCASE",
            (str(request.email).strip(),)
        ).fetchone()
    if user is None or not security.verify_password(
        request.password,
        user["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "access_token": security.create_access_token(user["id"]),
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "created_at": user["created_at"]
        }
    }


@app.get("/api/auth/me")
def current_user(user: dict = Depends(authenticated_user)):
    return user


@app.get("/api/conversations")
def list_conversations(user: dict = Depends(authenticated_user)):
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],)
        ).fetchall()
    return {"conversations": [dict(row) for row in rows]}


@app.post("/api/conversations", status_code=201)
def create_conversation(
    request: ConversationRequest,
    user: dict = Depends(authenticated_user)
):
    conversation_id = str(uuid.uuid4())
    now = utc_now()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO conversations(id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, user["id"], request.title.strip(), now, now)
        )
    return {
        "id": conversation_id,
        "title": request.title.strip(),
        "created_at": now,
        "updated_at": now
    }


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(
    conversation_id: str,
    user: dict = Depends(authenticated_user)
):
    with database.connect() as connection:
        owned_conversation(connection, conversation_id, user["id"])
        rows = connection.execute(
            "SELECT id, role, content, citations_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,)
        ).fetchall()
    return {
        "messages": [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "citations": database.decode_citations(row["citations_json"]),
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    }


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    user: dict = Depends(authenticated_user)
):
    with database.connect() as connection:
        owned_conversation(connection, conversation_id, user["id"])
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


@app.get("/api/documents")
def list_documents(user: dict = Depends(authenticated_user)):
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id, filename, media_type, status, created_at FROM documents "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],)
        ).fetchall()
    return {"documents": [dict(row) for row in rows]}


@app.get("/api/vector-index")
def vector_index_status(user: dict = Depends(authenticated_user)):
    try:
        with database.connect() as connection:
            return rag.get_index_status(connection, user["id"])
    except Exception as exc:
        logger.error("Vector index status could not be read.")
        raise HTTPException(
            status_code=503,
            detail="Vector index status is temporarily unavailable."
        ) from exc


@app.post(
    "/api/business-summary",
    response_model=BusinessSummaryResponse
)
def business_summary(
    request: BusinessSummaryRequest,
    user: dict = Depends(authenticated_user)
):
    try:
        summary = generate_business_summary(user["id"], request)
        return {"success": True, "summary": summary}
    except InvalidBusinessSummaryError as exc:
        logger.warning("Business summary generation returned invalid output.")
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an invalid business summary."
        ) from exc
    except BusinessSummaryUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Business summary generation is temporarily unavailable."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected business summary processing failure.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred."
        ) from exc


@app.post("/api/documents", status_code=201)
async def upload_document(
    file: UploadFile,
    user: dict = Depends(authenticated_user)
):
    filename = PurePath(file.filename or "").name.strip()
    extension = PurePath(filename).suffix.lower()
    if not filename or extension not in rag.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF, DOCX, or TXT document."
        )
    content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The upload exceeds 8 MB.")

    try:
        chunks = rag.chunk_pages(rag.parse_document(filename, content))
        if not chunks:
            raise ValueError("The document contains no extractable text.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        embedded_chunks = [
            (
                page,
                text,
                rag.validate_embedding(
                    rag.create_embedding(text, "RETRIEVAL_DOCUMENT")
                )
            )
            for page, text in chunks
        ]
        if len({len(embedding) for _, _, embedding in embedded_chunks}) != 1:
            raise ValueError("Embedding dimensions are inconsistent.")
    except Exception as exc:
        logger.error("Document ingestion failed.")
        raise HTTPException(
            status_code=503,
            detail="Document embedding is temporarily unavailable."
        ) from exc

    document_id = str(uuid.uuid4())
    now = utc_now()
    media_type = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain"
    }[extension]
    try:
        with database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO documents(id, user_id, filename, media_type, status, created_at) "
                "VALUES (?, ?, ?, ?, 'ready', ?)",
                (document_id, user["id"], filename[:255], media_type, now)
            )
            rag.store_embeddings(connection, document_id, embedded_chunks)
    except (ValueError, RuntimeError) as exc:
        logger.error("Document embeddings could not be stored.")
        raise HTTPException(
            status_code=503,
            detail="Document embeddings are incompatible with the vector index."
        ) from exc
    except sqlite3.Error as exc:
        logger.error("Document vector storage failed.")
        raise HTTPException(
            status_code=503,
            detail="Document storage is temporarily unavailable."
        ) from exc
    return {
        "id": document_id,
        "filename": filename[:255],
        "media_type": media_type,
        "status": "ready",
        "chunk_count": len(embedded_chunks),
        "created_at": now
    }


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str, user: dict = Depends(authenticated_user)):
    with database.connect() as connection:
        result = connection.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user["id"])
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
