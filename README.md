# Implement the Summary Generation Service

An authenticated FastAPI business assistant with conversation history, document ingestion, semantic retrieval, and source citations. The browser UI is served by the API, and Gemini is configured through environment variables.

## Architecture

```text
.
├── AI_service/
│   ├── api.py              # FastAPI application, lifecycle, middleware, non-chat routes
│   ├── routes/chat.py      # Authenticated chat HTTP endpoint
│   ├── controllers/        # HTTP error translation and response coordination
│   ├── services/           # Chat orchestration and Gemini model adapter
│   ├── schemas/chat.py     # Validated chat request/response and prompt assembly
│   ├── context/            # Verified business facts, model instructions, public knowledge
│   ├── dependencies.py     # Authentication and conversation ownership checks
│   ├── errors.py           # Sanitized chat service errors
│   ├── database.py         # SQLite schema and transactional connections
│   ├── rag.py              # Document parsing, chunking, embeddings, retrieval
│   ├── customer_context.py # User-scoped customer-record matching
│   ├── security.py         # Password hashing and signed bearer tokens
│   ├── settings.py         # Environment-based service configuration
│   ├── ai_service.py       # Compatibility facade and legacy CLI
│   └── static/             # Responsive browser client
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

The authenticated chat request flows from the router through the controller to
the chat service. The service retrieves user-owned documents and relevant public
business knowledge, builds a grounded prompt, calls the dedicated Gemini model
adapter, and persists the answer and citations. Provider credentials are read
from environment variables and are not returned to clients or written to logs.

```text
Browser UI (HTML/CSS/JS)
        │ same-origin bearer-token requests
        ▼
FastAPI ── auth + validation + SQLite-backed rate limits
   ├── SQLite: users, conversations, messages, documents, chunks, vectors, index configuration
  ├── RAG: PDF/DOCX/TXT parsing → overlapping chunks → Gemini embeddings
  │       → user-scoped document search + relevant public business knowledge
  │       → numbered document citations
   └── Gemini LLM: history + retrieved source context → grounded answer
```

### Database schema

| Table | Important fields and relations |
|---|---|
| `users` | `id`, unique case-insensitive `email`, PBKDF2 `password_hash`, `created_at` |
| `conversations` | `id`, `user_id → users.id`, `title`, `created_at`, `updated_at` |
| `messages` | `id`, `conversation_id → conversations.id`, `role`, `content`, `citations_json`, `created_at` |
| `documents` | `id`, `user_id → users.id`, `filename`, `media_type`, `status`, `created_at` |
| `chunks` | Stable `(document_id, chunk_index)` record key, `page_number`, `content`, `embedding_json` |
| `vector_index_state` | Embedding model, established vector dimension, cosine similarity metric |
| `rate_limit_buckets` | Shared SQLite fixed-window request counters keyed by client and route class |

SQLite provides durable storage and exact cosine vector search for a single-node
or modest dataset deployment. Each chunk upsert stores its text, embedding, and
document metadata together. Upserting the same document/chunk key updates its
record instead of adding a duplicate; deleting a document removes its chunks
through the foreign key cascade. Index initialization is idempotent
and never clears existing records. The configured Gemini embedding model and
the first stored vector's dimension are recorded and checked for later writes
and searches. Similarity retrieval is scoped to the authenticated user's
documents and accepts document ID, filename, and page-number filters through
the RAG service. `GET /api/vector-index` returns the index configuration and
the authenticated user's record count. This SQLite implementation performs an
exact scan of matching vectors; larger corpora or multi-node deployments
should migrate this operation to a dedicated vector database. The rate limiter
shares counters among workers that use the same SQLite file.

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
| `POST` | `/api/conversations/{id}/messages` | Yes | Ask a question with optional customer identifier and business/task context; returns answer and cited sources |
| `POST` | `/chat` | Yes | Send a required `message`; optionally continue a conversation with `conversation_id`; returns the answer, citations, and conversation ID |
| `POST` | `/api/business-summary` | Yes | Generate a structured summary from the caller's uploaded records and optional request-supplied business/task context |
| `POST` | `/api/communications/generate` | Yes | Generate a grounded email or short message with configurable purpose, tone, length, and language |
| `DELETE` | `/api/conversations/{id}` | Yes | Delete an owned conversation and its messages |
| `GET`, `POST` | `/api/documents` | Yes | List or upload PDF/DOCX/TXT files (8 MB max) |
| `DELETE` | `/api/documents/{id}` | Yes | Delete an owned document and its chunks |
| `GET` | `/api/vector-index` | Yes | Read vector index configuration and the user's record count |

Authenticated endpoints expect `Authorization: Bearer <access_token>`. The API rate limits each client IP to 10 auth requests/minute and 120 other API requests/minute. Configure trusted proxy handling at the ingress if deployed behind a reverse proxy.

### Email and message generation

`POST /api/communications/generate` requires the caller's bearer token.
`communication_type` is required and accepts `email` or `message`. Purpose
defaults to `customer_support` and also supports `follow_up`, `announcement`,
`invitation`, and `custom` (which requires `custom_purpose`). Tone defaults to
`professional` and supports `friendly`, `formal`, `polite`, `persuasive`, and
`empathetic`. Length defaults to `medium` and supports `short` and `long`.
Language defaults to English. Optional recipient name/context, sender name,
customer identifier, and additional instructions are validated and
length-limited. Email responses include a subject and body; message responses
contain a single `message` field with no email formatting.

Relevant public business knowledge is retrieved using the existing semantic
retriever. Customer-record excerpts are retrieved only when a customer
identifier is provided and are restricted to ready documents owned by the
authenticated user. Retrieved material is treated as data, not instructions.
Missing details are represented with a clear placeholder or a request for
clarification rather than invented facts. The endpoint generates content only;
it does not send communications.

Example request:

```json
{
  "communication_type": "email",
  "purpose": "follow_up",
  "tone": "professional",
  "length": "medium",
  "language": "English",
  "recipient_name": "John",
  "additional_instructions": "Follow up about the previous meeting."
}
```

Successful email responses contain `success: true` and a `content` object with
`subject` and `body`. Short-message responses contain `content.message`. Model
failures return HTTP 503; malformed model output returns HTTP 502.

### Prepared business knowledge

The authenticated message API retrieves semantically relevant entries from
the prepared public knowledge in `AI_service/context/business_context.py` using
the existing `AI_service/chat_api.py` retriever and combines
them with the user's document results. Bundled entries describe public facts
from the XICTEK Systems website. The legacy `/knowledge` write endpoint
requires `XICTEK_KNOWLEDGE_API_KEY`; admin-added records default to private and
are excluded from chat retrieval unless the administrator explicitly sets
`is_public: true`. Public entries are embedded lazily with the configured
Gemini embedding model and cached in process; query vectors must match the
document-vector dimensions. Retrieval returns the source, stable entry ID,
metadata, and cosine score and passes them into the existing grounded chat
prompt. `BUSINESS_KNOWLEDGE_TOP_K` (default `3`) and
`BUSINESS_KNOWLEDGE_MIN_RELEVANCE_SCORE` (default `0.35`) configure the result
count and relevance cutoff. This curated knowledge base and its embeddings are
in memory and do not persist across restarts. See
[RETRIEVAL_PIPELINE_REPORT.md](./RETRIEVAL_PIPELINE_REPORT.md) for the workflow,
tests, and limitations. Do not publish confidential business data.

### Optional business and task context

The message endpoint accepts context alongside the existing `message` field:

```json
{
  "message": "What should I report for this task?",
  "business_context": {
    "business_id": "business-1",
    "name": "Example Books",
    "details": "A small bookkeeping business."
  },
  "task_context": {
    "task_id": "task-1",
    "current_task": "Prepare annual report",
    "details": "Summarize the annual revenue.",
    "status": "in progress",
    "notes": "Use the uploaded report."
  }
}
```

Both context objects are optional, and omitted fields are allowed. The browser composer exposes the same fields; it sends the entered context with each message and clears it when switching conversations. Context values are request-provided, are not saved as task/business records, and are not independently verified. Retrieval remains restricted to the authenticated user's documents. Since documents currently have no business or task metadata, `business_id` and `task_id` are descriptive context only and do not filter retrieval.

### Customer records

The current schema has no dedicated customer table or customer model. Customer
records can be supplied through the authenticated user's existing PDF, DOCX,
or TXT uploads; no second customer store is created. Provide the customer's ID,
name, or contact in `customer_identifier` to find exact identifier matches in
the current user's document chunks. Those matched excerpts are prioritized
alongside normal semantic document retrieval and cited from their source
documents. The browser composer exposes this identifier field. A missing
match or field is reported as unavailable rather than inferred. Retrieval
remains isolated to the authenticated user's documents.

### Business summary

`POST /api/business-summary` requires the caller's bearer token. It uses only
the authenticated user's uploaded document chunks, optionally adds user-supplied
`business_context` and `task_context`, and can locate customer-related excerpts
when `customer_identifier` is provided. The summary is not persisted. The
current SQLite schema has no authoritative business profile, task, activity,
or customer tables, and uploaded documents have no business association; the
API therefore does not accept a `business_id` filter. `date_from` and
`date_to` (inclusive ISO dates) filter document upload timestamps only, not
dates mentioned inside document contents. At most the five most recent
document chunks plus up to five matching customer chunks are sent to the
configured Gemini model. Missing records and these limitations are reported.

Example request:

```json
{
  "date_from": "2026-10-01",
  "date_to": "2026-10-25",
  "customer_identifier": "C-104",
  "business_context": {
    "name": "Example Books",
    "details": "User-provided business description."
  },
  "task_context": {
    "current_task": "Prepare the monthly report",
    "status": "in progress"
  }
}
```

Successful responses contain `success: true` and a validated `summary` object
with `business_overview`, `key_activities`, `customer_updates`,
`task_updates`, `important_updates`, `operational_insights`, and
`data_limitations`. Model failures return HTTP 503; malformed model output
returns HTTP 502. See [BUSINESS_SUMMARY_REPORT.md](./BUSINESS_SUMMARY_REPORT.md)
for implementation details and test results.

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
python -m unittest AI_service.test_communication AI_service.test_business_summary AI_service.test_chat_api AI_service.test_rag_api -v
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
