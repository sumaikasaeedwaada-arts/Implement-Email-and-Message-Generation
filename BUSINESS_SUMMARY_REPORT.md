# Business Summary Generation Report

## Existing Architecture and Data Sources

The active application entry point is `AI_service.api:app`. Authenticated
requests use `authenticated_user`, and existing SQLite records are scoped by
`users.id`. The database stores users, conversations/messages, uploaded
documents, document chunks, vector-index state, and rate-limit buckets.

There are no persisted company-profile, operational-activity, task, or
customer tables. Business and task context fields already exist on chat
requests, but are request-provided and not independently verified. Customer
records are represented only where present in a user's uploaded documents;
the existing customer-context helper locates exact identifier matches in
user-owned chunks. Documents are user-scoped, not business-scoped.

The existing `AI_service/context/business_context.py` contains public,
prepared XICTEK Systems and HisabDo facts. Those organization-wide facts are
not user/business-owned records, so the personalized summary does not
automatically attribute them to an arbitrary authenticated account.

The project uses `AI_service.services.ai_model_service.generate_answer` as
its existing Gemini integration. The feature reuses it and the existing
database connection, settings, chat context schemas, customer matching helper,
and API authentication. No new persistence model, AI provider, or data source
was added.

## Implementation

- `AI_service/services/business_summary_service.py` collects a bounded set of
  the caller's most recent ready document chunks, optional exact-match customer
  excerpts, and optional request-supplied business/task context. Database
  queries always scope documents to the authenticated user's ID.
- Date filters apply to `documents.created_at` (upload time) and are inclusive
  by calendar date. They do not filter event dates inside document contents.
- At most five recent document chunks and five customer-match chunks are sent
  to Gemini. The response's limitations disclose when the document selection
  is truncated, when context is missing, and when the date range cannot
  establish event dates.
- The model receives structured JSON with explicit grounding, prompt-injection,
  privacy, recency, and missing-data instructions. Generated JSON is validated
  against a strict Pydantic model. Provider errors return a sanitized 503;
  empty or malformed output returns a sanitized 502. Logs do not include
  prompts, document content, customer details, provider responses, or
  credentials.
- If the user has no document content and provides no context, the service
  returns an empty, limitation-bearing summary without an unnecessary model
  call.
- The API does not accept `business_id`: this schema has no business ownership
  or mapping with which to verify such an identifier.

## API

`POST /api/business-summary` is bearer-authenticated and returns HTTP 200 for
complete or incomplete datasets, with missing data represented in the body.

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

Example successful response shape:

```json
{
  "success": true,
  "summary": {
    "business_overview": "Based on the supplied uploaded records...",
    "key_activities": [],
    "customer_updates": [],
    "task_updates": [],
    "important_updates": [],
    "operational_insights": [],
    "data_limitations": []
  }
}
```

HTTP 401 indicates missing/invalid authentication, 422 indicates invalid
request fields (including reversed dates), 502 indicates invalid model output,
503 indicates database/model unavailability, and 500 covers unexpected
internal failures. The existing API rate limiter covers this route.

## Configuration

No new environment variables are required. Set the existing
`GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_TIMEOUT_MS`, `AUTH_SECRET`, and
`DATABASE_PATH` values as documented in `.env.example`. The model and timeout
are inherited from the project's Gemini adapter settings.

## Tests and Verification

Automated tests are in `AI_service/test_business_summary.py`. They use a
temporary SQLite database and mock `generate_answer`; no live Gemini API calls
are made. Coverage includes document-grounded facts, customer matching, empty
data, upload-date filtering, cross-user isolation, authentication, request
validation, database errors, model timeouts, and malformed AI output.

**Test execution results:** `Ran 56 tests in 18.302s` — `OK`. The combined run
included `AI_service.test_business_summary`, `AI_service.test_chat_api`, and
`AI_service.test_rag_api`. Python compilation of all five changed Python files
also completed successfully, and editor diagnostics reported no errors in
those files. These tests mocked model calls; no live Gemini API request was
made. The terminal emitted an existing Starlette/httpx `TestClient`
deprecation warning.

The configured local database path was checked in read-only mode and no
database file was present. Therefore, endpoint behavior was verified against
temporary SQLite test fixtures, not live company/customer records. A live
business summary with the configured database and Gemini provider remains
unverified.

Run the application and tests with:

```powershell
uvicorn AI_service.api:app --reload --env-file .env
python -m unittest AI_service.test_business_summary AI_service.test_chat_api AI_service.test_rag_api -v
```

No screenshot was captured: verification uses FastAPI's automated `TestClient`
and mocked model calls rather than a live browser/model session.

## Limitations and Recommended Improvements

- Add authoritative tenant/business ownership and persisted business, task,
  customer, and operational-activity records before offering business-ID
  filtering or claiming those records are verified.
- Add document-to-business attribution before allowing business-specific
  document isolation; current access is scoped to users.
- Improve record selection beyond the fixed recent-chunk cap (for example,
  date-aware ranking or semantic retrieval over a business-attributed corpus)
  while preserving bounded prompts and clear truncation metadata.
- Model-produced summaries remain generative. Pydantic validation guarantees
  response shape, not factual entailment; source citations and human review
  are appropriate for consequential reporting.
