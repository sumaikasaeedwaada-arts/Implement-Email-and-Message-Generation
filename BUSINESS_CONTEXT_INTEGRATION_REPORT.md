# Business Context Integration Report

## Existing Implementation and Issues

- The documented Docker entry point is `AI_service.api:app`. Its authenticated message route already performed user-scoped SQLite document retrieval through `AI_service.rag.search_chunks` and sent retrieved excerpts plus history to Gemini.
- Verified XICTEK Systems and HisabDo facts were stored as a static text constant in `AI_service.ai_service.BUSINESS_CONTEXT`. The production RAG prompt sent that entire block on every question, even when unrelated.
- A prepared, embedding-backed business knowledge source already existed in `AI_service.chat_api.knowledge_base` and `search_knowledge`, but the authenticated API did not use it. The legacy `/chat` prompt also included the entire static business block in addition to retrieved knowledge.
- The database has no business/task tables or business-to-document attribution. `business_context` and `task_context` in the authenticated message request are optional, bounded, request-scoped values; they are not authoritative records and do not filter document access.
- The legacy `/chat` endpoint is unauthenticated while its admin knowledge-write endpoint accepts arbitrary knowledge. Without visibility control, private admin-added facts could be retrieved into public chat prompts.

## Changes Made

- Integrated the existing semantic `chat_api.search_knowledge` retriever into the authenticated message flow. Only its relevant hits are added to the prompt alongside existing user-owned document results.
- Changed the structured prompt to include selected `retrieved_business_context`, request-provided business/task context, bounded conversation history, document excerpts, and the current question. Exact duplicate business excerpts already present in documents are removed; malformed business hits are skipped.
- Business retrieval failures now log a warning and continue with available document/user context. Existing document retrieval failures retain their sanitized 503 behavior. Existing response fields and document citations are unchanged.
- Removed the always-on static business profile from the legacy `/chat` prompt; it now uses its already-retrieved knowledge results.
- Strengthened system and request instructions: prefer relevant evidence, do not invent facts, admit missing information, identify source conflicts, treat retrieved material as data rather than instructions, and do not disclose confidential information without authorization.
- Added `is_public` visibility to the legacy prepared knowledge source. Bundled website facts are marked public; admin-added knowledge defaults to private and enters shared retrieval only when explicitly published. The admin write API remains protected by `XICTEK_KNOWLEDGE_API_KEY`.
- Kept root-level module/test mirrors synchronized and updated the README and this report.

## Context Flow

1. The production endpoint authenticates the caller and verifies ownership of the conversation.
2. It searches SQLite document chunks using the authenticated user's ID, preserving existing relevance filtering and document citations.
3. It searches the existing prepared business knowledge using the same user query and the existing semantic relevance threshold. Only entries marked public are embedded/retrieved from this shared source.
4. It combines relevant business hits, document excerpts, request-scoped business/task fields, bounded history, and the question as structured JSON. Duplicate excerpts are omitted.
5. Gemini receives the JSON prompt and the strengthened system instructions. A business retriever error omits those hits rather than blocking a response from available documents; a document retriever error remains a service error.

The legacy `/chat` endpoint also uses selected prepared knowledge, but it remains a separate, unauthenticated legacy API. Its shared retrieval is limited to explicitly public facts. Authenticated SQLite documents are never queried by that endpoint.

## Tests and Actual Results

Added/updated coverage for:

- Business-only prepared context entering the production prompt.
- Combined business context, task context, and retrieved document excerpts.
- Document-only responses when business retrieval fails.
- Empty sources and irrelevant document hits.
- Duplicate and malformed business excerpts.
- Missing facts, conflict handling, and non-fabrication prompt requirements.
- Authentication before business retrieval and existing per-user document isolation.
- Exclusion of private admin knowledge from public chat, private-by-default writes, and explicit public publication.
- Existing citations, upload, history, and legacy chat behavior.

Executed:

```text
python -m unittest AI_service.test_chat_api AI_service.test_rag_api -v
Ran 34 tests in 7.604s
OK
```

Provider calls were mocked in these tests; no live Gemini generation or embedding request was made. The project run and test instructions remain in `README.md`.

## Limitations and Recommendations

- The prepared business knowledge base and its vector store are in-memory, so admin-added content is lost on restart. Move it to a durable store with explicit tenant/business ownership and visibility metadata.
- This repository has no authoritative business/task records. Request-provided fields remain user-supplied and are not verified or persisted as business records.
- `is_public: true` intentionally makes admin knowledge available through the legacy public chat and shared authenticated business retrieval. Administrators must not publish confidential facts.
- Documents are isolated per user, not per business. Per-business document retrieval requires business attribution at upload plus ownership checks during search; caller-supplied IDs alone are not sufficient.
- Prompt instructions reduce unsupported answers but cannot guarantee model compliance. Continue testing source conflicts and authorization when a durable, multi-business source is introduced.
- The legacy `/chat`, `/ask`, and CLI paths remain separate from the SQLite RAG service; the latter two retain their existing static-context behavior.
