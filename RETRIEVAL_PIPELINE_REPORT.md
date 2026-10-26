# Retrieval Pipeline Report

## Existing architecture

The deployed FastAPI application is `AI_service.api:app`. Its authenticated
chat flow is routed through `AI_service.services.chat_service`: it retrieves
the current user's document chunks from SQLite, retrieves prepared business
knowledge through `AI_service.chat_api`, assembles both source sets into the
existing grounded prompt, calls the Gemini response service, and persists the
answer and document citations.

Two retrieval sources are in use:

- User-uploaded PDF, DOCX, and TXT documents are chunked and stored with their
  Gemini embeddings in SQLite (`chunks` joined to user-owned `documents`).
  Retrieval uses cosine similarity and an authenticated `user_id` predicate.
- Prepared public business knowledge comes from
  `AI_service/context/business_context.py`. Its vectors are generated lazily
  and cached in process memory. Private admin entries are excluded. The older
  `/chat` endpoint in `AI_service.chat_api` also uses this same retriever.

The embedding model defaults to `gemini-embedding-001`, configurable through
`GEMINI_EMBEDDING_MODEL`. Gemini is also the response-generation provider; the
chat model defaults to `gemini-3.5-flash-lite` through `GEMINI_MODEL`. Both
services require `GEMINI_API_KEY`. User-document vectors are persisted in
SQLite; prepared business knowledge is currently a curated in-code list with
an in-memory vector cache, not a separate vector database.

## Problems addressed

- Business knowledge vectors were rebuilt whenever the in-memory list was
  empty, with no cache validation when its public contents changed.
- Query and stored vectors lacked validation for empty, non-finite, zero, or
  incompatible-dimensional vectors.
- Search had a fixed result count and threshold and offered no supported
  metadata filters.
- The retriever omitted structured metadata from results, and the authenticated
  RAG prompt dropped business record IDs, metadata, and similarity scores.
- Empty public indexes should return no evidence without making unnecessary
  embedding calls.

## Implementation and files

- `AI_service/chat_api.py`: validates vectors and queries, keeps the public
  knowledge index cache synchronized with public records, skips malformed or
  duplicate-ID entries, enforces consistent dimensions and public visibility,
  and returns IDs, text, metadata, source, and cosine scores. Search supports
  configurable top-k and exact filters for `id`, `title`, and `source`.
- `AI_service/settings.py`: adds environment configuration for business
  knowledge top-k and the minimum cosine score.
- `AI_service/schemas/chat.py`: preserves business record IDs, scores, and
  metadata in the authenticated RAG prompt.
- `AI_service/test_chat_api.py`: covers cache reuse, filtering, top-k, query
  normalization, empty indexes, invalid vectors, and sanitized retrieval
  failures.
- `AI_service/test_rag_api.py`: verifies business metadata reaches the existing
  prompt without disrupting the document-chat response.
- `README.md`: documents retrieval settings and points to this report.

No new dependencies, APIs, or second RAG pipeline were introduced.

## Query-to-RAG workflow

1. The API validates the question; the retriever strips surrounding whitespace
   for embedding only and rejects blank queries.
2. On first search or after public knowledge changes, the retriever embeds
   valid public entries with `RETRIEVAL_DOCUMENT`. It keeps stable entry IDs,
   title/content/source metadata, and vectors together, and atomically replaces
   the in-memory cache only after a successful build.
3. The query is embedded with `RETRIEVAL_QUERY`. Invalid or incompatible
   vectors fail explicitly; empty knowledge returns an empty result list.
4. Supported metadata filters and the public-only rule are applied, cosine
   similarity scores are thresholded, and the highest-scoring top-k records are
   returned.
5. The authenticated chat service passes retrieved prepared business records
   alongside user-scoped document excerpts to the existing prompt builder.
   The prompt retains business IDs, scores, metadata, title, source, and text.
   User-document citations and the existing API response format remain intact.
6. A prepared-knowledge retrieval failure in authenticated chat is logged
   without sensitive provider details and does not prevent answering from
   available user documents or request context. The legacy `/chat` endpoint
   returns its existing sanitized service-unavailable response on retrieval
   failure.

## Tests and validation

Run the focused suite with:

```powershell
python -m unittest AI_service.test_chat_api AI_service.test_rag_api -v
```

Executed:

```text
python -m unittest AI_service.test_chat_api AI_service.test_rag_api -v
Ran 44 tests in 8.892s
OK
```

All 44 tests passed, including an end-to-end test of actual business-vector
search and handoff into the authenticated RAG prompt. Tests mock external
Gemini requests; they do not require a live API key.

`python -m compileall -q AI_service` also completed successfully.

## Setup and limitations

Install the existing dependencies and configure `GEMINI_API_KEY` as described
in the [README](./README.md). Start the service from PowerShell with:

```powershell
uvicorn AI_service.api:app --reload --env-file .env
```

Run the retrieval and RAG tests with:

```powershell
python -m unittest AI_service.test_chat_api AI_service.test_rag_api -v
```

Optional retrieval settings:

```text
BUSINESS_KNOWLEDGE_TOP_K=3
BUSINESS_KNOWLEDGE_MIN_RELEVANCE_SCORE=0.35
```

The cosine threshold is used to exclude low-scoring prepared business results.
User-owned document search continues to use its existing threshold and
user-scoped SQLite index. Prepared entries and their embeddings are not
persistent, and the cache is process-local; each worker may generate its own
copy at first use. Prepared business entries currently have no tenant/business
ownership field, so only records explicitly marked public are eligible.
Configure a durable managed vector index only if the knowledge base grows or
must be updated independently of application code.
