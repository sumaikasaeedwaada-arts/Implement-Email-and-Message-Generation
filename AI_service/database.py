import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from AI_service import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_user_updated
    ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conversation_created
    ON messages(conversation_id, created_at, id);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_user_created
    ON documents(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_document
    ON chunks(document_id, chunk_index);
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    bucket_key TEXT PRIMARY KEY,
    window_started REAL NOT NULL,
    request_count INTEGER NOT NULL
);
"""


def database_path() -> str:
    return os.getenv("DATABASE_PATH", settings.DATABASE_PATH)


@contextmanager
def connect():
    path = database_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    if path != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)


def row_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def decode_citations(value: str) -> list[dict]:
    citations = json.loads(value)
    return citations if isinstance(citations, list) else []
