# Email and Message Generation

## Existing architecture and findings

The application is an authenticated FastAPI service in `AI_service/api.py`.
Protected routes use the shared bearer-token dependency, and SQLite document
queries are scoped to the authenticated user. The existing model adapter in
`AI_service/services/ai_model_service.py` supplies the configured Gemini model,
the shared business-assistant system prompt, and the configured provider
timeout.

The repository already includes a prepared public business knowledge base and
semantic retriever in `AI_service/chat_api.py`, plus a user-scoped document
chunk store and exact customer-identifier matching in
`AI_service/customer_context.py`. It does not have a dedicated structured
email/message generation endpoint or normalized customer/contact tables.
Existing chat output is free-form, so it does not guarantee email subject/body
fields.

## Implementation

- Added `POST /api/communications/generate`, protected by existing
  authentication and the API's normal rate limiting.
- Added Pydantic request/output schemas with bounded text fields and enum
  validation. `communication_type` is required; remaining options have
  defaults. A custom purpose requires a description.
- Added a reusable communication service that retrieves public business
  knowledge through the existing retriever, looks up customer excerpts only
  when the request includes a customer identifier, and uses the existing
  `ai_service.generate_answer` model integration.
- Customer excerpts are retrieved only from ready documents belonging to the
  authenticated user's account. Public business entries are passed on only
  when explicitly marked public.
- The model prompt treats retrieved text as data, limits claims to supplied
  context, omits unsupplied sender signatures, and requests placeholders or
  clarification rather than invented facts.
- Model output is parsed and validated before returning. Provider/context
  failures are sanitized as HTTP 503; malformed model output is HTTP 502;
  unexpected processing errors return a generic HTTP 500 response.
- Email responses contain `subject` and `body`; message responses contain one
  `message` field without email formatting.

### Files changed

- `AI_service/schemas/communication.py` — request and response schemas.
- `AI_service/services/communication_service.py` — context retrieval, prompt
  construction, model call, and output validation.
- `AI_service/api.py` — authenticated API route and error translation.
- `AI_service/test_communication.py` — mocked API and context-isolation tests.
- `README.md` — API documentation and test command.
- `EMAIL_MESSAGE_GENERATION_REPORT.md` — this report.

## Supported options

| Option | Supported values | Default |
|---|---|---|
| Communication type | `email`, `message` | Required |
| Purpose | `customer_support`, `follow_up`, `announcement`, `invitation`, `custom` | `customer_support` |
| Tone | `professional`, `friendly`, `formal`, `polite`, `persuasive`, `empathetic` | `professional` |
| Length | `short`, `medium`, `long` | `medium` |
| Language | Non-blank text, up to 100 characters | `English` |

`custom_purpose` is required only when `purpose` is `custom`. Optional fields
include `recipient_name`, `recipient_context`, `customer_identifier`,
`additional_instructions`, and `sender_name`. Request fields are validated
and capped at reasonable maximum lengths.

## API examples

Request:

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

Email response:

```json
{
  "success": true,
  "content": {
    "subject": "Following Up on Our Previous Meeting",
    "body": "Hello John, ... "
  }
}
```

Message requests use `"communication_type": "message"`; their response
`content` has the shape `{"message": "..."}` and has no subject or email body.

## Business context and privacy

Relevant public entries come from the existing semantic business retriever.
The existing model integration also applies the shared business-assistant
system prompt. Customer excerpts are fetched only if `customer_identifier` is
provided, using the existing user-scoped matching helper; no other user's
records are included. Customer-record chunks are not retrieved merely to
personalize a general communication. The service avoids including document
filenames and other unnecessary metadata in its prompt.

The current data store has uploaded documents and text chunks, rather than
normalized customer records. Exact customer-identifier matching can therefore
miss records that use different wording or identifiers. User-provided
recipient details are treated as user input, not independently verified
business facts. Generated content should be reviewed before sending; this
endpoint does not send communications.

## Tests and verification

Executed:

```powershell
python -m unittest AI_service.test_communication AI_service.test_business_summary AI_service.test_chat_api AI_service.test_rag_api -v
```

Result: **65 tests passed** in 45.795 seconds (`OK`).

Communication tests mock model responses and retrieval where appropriate.
They cover structured email and message output, tone/length/purpose/language,
defaults and custom purpose, missing recipient/context facts, public-context
filtering, customer-record ownership, validation/authentication, and sanitized
provider/output failures. Existing chat, business-summary, and RAG API tests
were also included. These are **mocked tests**, not a live
Gemini integration test; no request was sent to the live model provider.

Pylance syntax checks found no syntax errors in the new Python modules, and
Pylance diagnostics for the new schema, service, and test files were empty.

## Configuration and running

The service reuses the configured `GEMINI_MODEL` (default
`gemini-3.5-flash-lite`) and `GEMINI_TIMEOUT_MS` (default `30000`). Provide
`GEMINI_API_KEY` through the environment for real Gemini generation and
business-knowledge embeddings. Production deployments should also configure
a stable, secret `AUTH_SECRET` and a durable `DATABASE_PATH`.

Start the application using the existing project setup:

```powershell
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key"
$env:AUTH_SECRET = "a-long-random-secret"
$env:DATABASE_PATH = "data/assistant.db"
uvicorn AI_service.api:app --reload
```

Run the API and regression tests:

```powershell
python -m unittest AI_service.test_communication AI_service.test_business_summary AI_service.test_chat_api AI_service.test_customer_insights AI_service.test_rag_api -v
```

## Remaining limitations

- No live Gemini integration test was run; actual output quality, latency, and
  provider availability depend on runtime credentials and service conditions.
- Retrieval of relevant public business context uses the existing embedding
  retriever. If that retrieval fails, generation continues without those
  retrieved facts and relies on the shared system prompt and supplied request
  context.
- Prompt constraints and schema validation reduce unsupported claims and
  formatting errors but cannot mathematically guarantee that a language model
  will never hallucinate. Review generated content before sending.
- No screenshot was captured: this is a backend API change, and the available
  verification was through mocked FastAPI `TestClient` requests rather than a
  browser-based live-provider session.
