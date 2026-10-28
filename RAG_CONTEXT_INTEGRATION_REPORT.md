# RAG and Task Context Integration Report

## Existing architecture

- The documented and Dockerized entry point is `AI_service.api:app` in
  [`AI_service/api.py`](./AI_service/api.py). Root-level `api.py`, `rag.py`,
  `ai_service.py`, and the RAG test file mirror their `AI_service/` package
  counterparts; both copies of changed modules are kept synchronized.
- The primary API uses FastAPI, bearer-token authentication, and SQLite via
  [`AI_service/database.py`](./AI_service/database.py). Its schema stores users,
  conversations, messages, uploaded documents, parsed chunks, JSON embeddings,
  and rate limits. There are no business or task tables or data-access
  services.
- The primary retrieval implementation is
  [`AI_service/rag.py`](./AI_service/rag.py). It parses PDF, DOCX, and TXT
  uploads; chunks text; creates Gemini embeddings using
  `GEMINI_EMBEDDING_MODEL`; and calculates cosine similarity over SQLite
  vectors. Retrieval filters documents by the authenticated `user_id`, then
  applies the configured minimum score and result limit.
- [`AI_service/ai_service.py`](./AI_service/ai_service.py) configures Gemini
  generation and contains the existing fixed `BUSINESS_CONTEXT` describing
  XICTEK Systems and HisabDo. Before this change, the primary chat prompt did
  not include that verified business context.
- The primary API fetches a bounded history of persisted conversation
  messages. The frontend in [`AI_service/static/`](./AI_service/static/)
  submits to this authenticated API.
- [`AI_service/chat_api.py`](./AI_service/chat_api.py) is a separate legacy
  `/chat` application. It has a small in-memory knowledge/vector store and
  accepts free-form user, business, customer, task, and notes fields. Its
  conversation history is also in memory. It is not the Dockerized application
  entry point and is not connected to the SQLite document RAG pipeline.
- [`chatbot.py`](./chatbot.py) starts a separate legacy CLI. Dependencies and
  the documented local/Docker launch instructions are in
  [`requirements.txt`](./requirements.txt), [`README.md`](./README.md),
  [`Dockerfile`](./Dockerfile), and [`docker-compose.yml`](./docker-compose.yml).

## Problems identified

1. The primary authenticated message API combined retrieved document excerpts
   and conversation history, but omitted the existing verified
   `BUSINESS_CONTEXT` and had no fields for current task/business context.
2. There is no authoritative business or task data source in the primary
   database. The legacy endpoint's free-form context fields do not constitute
   a stored task-management system and are not available to the primary API.
3. The primary `documents` and `chunks` schema has no business or task
   attribution. Retrieval is user-scoped, not business-scoped. An identifier
   supplied by a caller cannot safely be used to filter existing documents.
4. Malformed stored embedding JSON or values previously caused document search
   to fail as a whole, returning a service error instead of skipping the
   unusable vector.
5. The generation instructions did not clearly separate verified static
   business facts, caller-provided context, retrieved evidence, and historical
   conversation content, or explain what to do when these sources conflict.

## Changes made

- Added optional, bounded `business_context` and `task_context` request models
  to the existing authenticated message endpoint in
  [`AI_service/api.py`](./AI_service/api.py). They include business/task IDs
  (descriptive only), business name/details/notes, and current task,
  details/status/notes. Existing clients that send only `message` remain
  compatible.
- Added a structured prompt builder that combines the existing verified
  business context, non-empty request-provided context, the existing bounded
  conversation history, the current question, and only the retrieved
  document chunks that pass relevance filtering. Existing citation numbering
  and API response fields are unchanged.
- Strengthened the shared Gemini system prompt and per-request instructions:
  caller-supplied context is not independently verified; history is for
  relevant references rather than authoritative facts; retrieved chunks are
  cited; missing, stale, or conflicting information must be acknowledged or
  clarified; and document/business association must not be assumed.
- Updated the browser composer to optionally collect and submit business and
  task fields. They are sent with each message, not persisted as task/business
  records, and cleared when changing conversations.
- Hardened vector retrieval to log and skip malformed stored vectors instead
  of aborting the complete search.
- Kept root-level module mirrors synchronized with the production package.
- Documented the API payload and context limitations in
  [`README.md`](./README.md), and added integration tests in
  [`AI_service/test_rag_api.py`](./AI_service/test_rag_api.py).

## How context is combined

For each authenticated message, the existing endpoint:

1. Verifies the conversation belongs to the authenticated user.
2. Loads its bounded persisted history.
3. Embeds the current question and searches only that user's uploaded document
   chunks. Existing similarity thresholding and result limits remain in
   effect.
4. Builds a JSON prompt with verified static company context, optional
   caller-provided business and task context, history, relevant document
   excerpts and citation indices, and the current question.
5. Calls the existing Gemini generation service. It saves the user message and
   assistant response only after successful generation, returning the same
   response shape and document citations as before.

The business/task fields are request-scoped. They are not looked up, persisted,
or treated as verified records. `business_id` and `task_id` currently do not
filter retrieval because stored documents have no corresponding metadata.

## Tests and validation

Executed on the configured Python 3.14.8 environment:

```text
python -m unittest AI_service.test_rag_api AI_service.test_chat_api -v
Ran 29 tests in 6.207s
OK
```

The tests cover existing document ingestion/retrieval/citations and legacy
chat behavior, business-only and task-only contexts, combined document plus
business/task context, missing context and empty retrieval, irrelevant
retrieval, current context versus stale history, malformed stored vectors,
invalid context validation, isolation between different users, retrieval
failures, and LLM failures without persisting a failed exchange.

Also executed:

- `node --check AI_service\static\app.js` — passed.
- `git -c core.whitespace=cr-at-eol --no-pager diff --check` — passed.
- Pylance Python syntax checks for the changed API, RAG, and test modules —
  no syntax errors reported.

No dedicated lint configuration or lint command was present in the repository.
Pylance workspace diagnostics did report unresolved imports such as FastAPI,
Pydantic, and `google.genai` in its selected editor environment; the configured
terminal Python environment successfully imported the application and ran the
tests. The editor environment should be set to an interpreter with the
dependencies from `requirements.txt` installed.

## Known limitations and recommendations

- Task and business context is currently caller-provided, not fetched from an
  authoritative application data source. The model instructions reduce
  unsupported claims but cannot verify the freshness or correctness of those
  values.
- Documents are isolated across authenticated users, but not across multiple
  businesses belonging to the same user. Per-business filtering cannot be
  implemented safely until business ownership metadata is captured when
  documents are uploaded and checked during retrieval. The supplied IDs are
  not a substitute for that authorization/attribution.
- The primary API's static business context is existing embedded text, not
  versioned or dynamically retrieved content. Its facts are not represented as
  document citations in message responses.
- The separate legacy `/chat` service retains its own in-memory retrieval and
  conversation behavior; this integration does not merge it into the
  authenticated production API.
- Prompt instructions do not guarantee factual compliance from a live model.
  Provider behavior was mocked in tests; no live Gemini generation or
  embedding request was made.
- Browser context fields were syntax-checked but not exercised in an automated
  browser test; the repository has no configured frontend test runner.
- Recommended next steps are to connect an authorized business/task source,
  associate uploaded documents with business identifiers, enforce those
  ownership filters during retrieval, define freshness/conflict rules for
  task status, and test same-user cross-business isolation once those data
  sources exist.
