# AI Business Assistant

An authenticated FastAPI business assistant with conversation history, document ingestion, semantic retrieval, and source citations. The browser UI is served by the API, and Gemini is configured through environment variables.

## Architecture

```text
.
├── AI_service/
│   ├── api.py              # New authenticated FastAPI/RAG application
│   ├── database.py         # SQLite schema and transactional connections
│   ├── rag.py              # Parsers, chunking, embeddings, vector retrieval
│   ├── security.py         # Password hashing and signed bearer tokens
│   ├── settings.py         # Environment-based service configuration
│   ├── ai_service.py       # Gemini chat client and legacy CLI
│   ├── app.py              # Legacy /ask endpoint
│   ├── chat_api.py         # Legacy /chat endpoint
│   └── static/             # Responsive browser client
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

```text
Browser UI (HTML/CSS/JS)
        │ same-origin bearer-token requests
        ▼
FastAPI ── auth + validation + SQLite-backed rate limits
   ├── SQLite: users, conversations, messages, documents, chunks, vectors
   ├── RAG: PDF/DOCX/TXT parsing → overlapping chunks → Gemini embeddings
   │       → user-scoped cosine similarity search → numbered citations
   └── Gemini LLM: history + retrieved source context → grounded answer
```

### Database schema

| Table | Important fields and relations |
|---|---|
| `users` | `id`, unique case-insensitive `email`, PBKDF2 `password_hash`, `created_at` |
| `conversations` | `id`, `user_id → users.id`, `title`, `created_at`, `updated_at` |
| `messages` | `id`, `conversation_id → conversations.id`, `role`, `content`, `citations_json`, `created_at` |
| `documents` | `id`, `user_id → users.id`, `filename`, `media_type`, `status`, `created_at` |
| `chunks` | `document_id → documents.id`, `chunk_index`, `page_number`, `content`, `embedding_json` |
| `rate_limit_buckets` | Shared SQLite fixed-window request counters keyed by client and route class |

SQLite provides durable storage and exact cosine vector search for a single-node or modest dataset deployment. It scans the requesting user's chunks; larger corpora or multi-node deployments should migrate this table/search operation to a managed vector database. The rate limiter shares counters among workers that use the same SQLite file.

## API

Interactive OpenAPI documentation is available at `/docs`; the schema is at `/openapi.json`.

| Method | Path | Auth | Purpose |
|---|---|---:|---|
| `GET` | `/api/health` | No | Health probe |
| `POST` | `/api/auth/register` | No | Register (password minimum 10 characters) |
| `POST` | `/api/auth/login` | No | Login; returns a bearer access token |
| `GET` | `/api/auth/me` | Yes | Current account |
| `GET`, `POST` | `/api/conversations` | Yes | List or create conversations |
| `GET` | `/api/conversations/{id}/messages` | Yes | Read owned conversation history |
| `POST` | `/api/conversations/{id}/messages` | Yes | Ask a question; returns answer and cited sources |
| `DELETE` | `/api/conversations/{id}` | Yes | Delete an owned conversation and its messages |
| `GET`, `POST` | `/api/documents` | Yes | List or upload PDF/DOCX/TXT files (8 MB max) |
| `DELETE` | `/api/documents/{id}` | Yes | Delete an owned document and its chunks |

Authenticated endpoints expect `Authorization: Bearer <access_token>`. The API rate limits each client IP to 10 auth requests/minute and 120 other API requests/minute. Configure trusted proxy handling at the ingress if deployed behind a reverse proxy.

## Run locally

Requires Python 3.11+ and a Gemini API key for actual chat and embedding requests.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` and a long, random `AUTH_SECRET`. Never commit `.env`; it is excluded by `.gitignore`. Set environment variables in the current PowerShell session and start the app:

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:AUTH_SECRET = "a-long-random-secret"
$env:DATABASE_PATH = "data/assistant.db"
uvicorn AI_service.api:app --reload --env-file .env
```

Open <http://127.0.0.1:8000>. Create an account, upload a file, and ask a question whose answer appears in it. The service returns an explicit 503 when the Gemini key/provider is unavailable; it does not silently generate embeddings or fabricated answers.

## Tests

```powershell
python -m unittest AI_service.test_chat_api AI_service.test_rag_api -v
```

Tests mock model calls and use a temporary SQLite database; they do not require a real provider key.

## Docker deployment

Create a local `.env` based on `.env.example`, set the actual key and a cryptographically random `AUTH_SECRET`, then run:

```powershell
docker compose up --build -d
docker compose logs -f assistant
```

The database persists in the `assistant-data` volume. Back up that volume regularly. Terminate TLS at a trusted reverse proxy, restrict direct access to the container port, use a stable high-entropy `AUTH_SECRET` shared by all workers, rotate provider credentials via environment/secret management, and monitor `/api/health` and application logs. Configure proxy IP forwarding only for a trusted ingress. Do not expose the SQLite volume publicly.

## Security and operational notes

- Passwords are PBKDF2-HMAC-SHA256 hashes with per-user random salts; bearer tokens are HMAC-signed and expire after one hour by default.
- Set `AUTH_SECRET` in deployment. When omitted in local development, a per-process random key is used and a warning is logged; tokens then do not survive restarts or work across workers.
- Conversation and document queries are scoped to the authenticated user. Uploaded files are parsed in memory and their text/vector data is persisted; original upload bytes are not retained.
- Upload size, supported suffix, empty content, chunk count, and request field lengths are bounded and validated. Avoid logging uploaded document bodies, passwords, tokens, or provider credentials.
- SQLite-backed exact vector search is appropriate as a straightforward durable baseline, not a replacement for ANN indexes at large scale.
