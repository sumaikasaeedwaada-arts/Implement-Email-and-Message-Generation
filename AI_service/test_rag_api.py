import json
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from AI_service import api, rag, security


class RagUnitTests(unittest.TestCase):
    def test_text_chunking_keeps_overlap_and_content(self):
        chunks = rag.chunk_pages([(None, "abcdefghij " * 8)], max_chars=30, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(content) <= 30 for _, content in chunks))
        self.assertIn(chunks[0][1][-5:], chunks[1][1])

    def test_txt_parser_and_invalid_extension(self):
        self.assertEqual(
            rag.parse_document("notes.txt", b"hello"),
            [(None, "hello")]
        )
        with self.assertRaises(ValueError):
            rag.parse_document("notes.exe", b"hello")

    def test_docx_and_pdf_parsers(self):
        from docx import Document
        from pypdf import PdfWriter

        word_file = BytesIO()
        word_document = Document()
        word_document.add_paragraph("Quarterly results")
        word_document.save(word_file)
        parsed_docx = rag.parse_document("results.docx", word_file.getvalue())
        self.assertIn("Quarterly results", parsed_docx[0][1])

        pdf_file = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(pdf_file)
        parsed_pdf = rag.parse_document("results.pdf", pdf_file.getvalue())
        self.assertEqual(parsed_pdf, [(1, "")])

    def test_password_and_signed_token_validation(self):
        encoded = security.hash_password("correct horse battery staple")
        self.assertTrue(security.verify_password("correct horse battery staple", encoded))
        self.assertFalse(security.verify_password("incorrect", encoded))
        token = security.create_access_token(17)
        self.assertEqual(security.read_access_token(token), 17)
        self.assertIsNone(security.read_access_token(token + "x"))


class RagApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_directory.name, "test.sqlite")
        self.env = patch.dict(os.environ, {
            "DATABASE_PATH": self.db_path,
            "AUTH_SECRET": "test-auth-secret-not-for-production",
            "GEMINI_API_KEY": "test-provider-key"
        })
        self.env.start()
        self.business_search = patch.object(
            api.chat_api,
            "search_knowledge",
            return_value=[]
        )
        self.business_search.start()
        api.database.initialize_database()
        self.client = TestClient(api.app)
        self.client.__enter__()
        result = self.client.post("/api/auth/register", json={
            "email": "owner@example.com",
            "password": "a-secure-test-password"
        })
        self.assertEqual(result.status_code, 201)
        self.headers = {"Authorization": f"Bearer {result.json()['access_token']}"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.business_search.stop()
        self.env.stop()
        self.temp_directory.cleanup()

    def test_authentication_and_duplicate_registration(self):
        duplicate = self.client.post("/api/auth/register", json={
            "email": "OWNER@example.com",
            "password": "a-secure-test-password"
        })
        self.assertEqual(duplicate.status_code, 409)
        profile = self.client.get("/api/auth/me", headers=self.headers)
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["email"], "owner@example.com")
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_chat_endpoint_returns_structured_reply_and_persists_history(self):
        with patch.object(
            api.ai_service,
            "generate_answer",
            return_value="The answer is available."
        ) as generate:
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"message": "  What is the answer?  "}
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["response"], "The answer is available.")
        self.assertTrue(payload["conversation_id"])
        self.assertEqual(payload["citations"], [])
        self.assertTrue(payload["created_at"])
        self.assertEqual(
            json.loads(generate.call_args.args[0])["current_question"],
            "What is the answer?"
        )
        history = self.client.get(
            f"/api/conversations/{payload['conversation_id']}/messages",
            headers=self.headers
        )
        self.assertEqual(len(history.json()["messages"]), 2)
        self.assertEqual(
            history.json()["messages"][0]["content"],
            "What is the answer?"
        )

    def test_chat_endpoint_validates_messages_and_requires_authentication(self):
        missing_message = self.client.post(
            "/chat",
            headers=self.headers,
            json={}
        )
        blank_message = self.client.post(
            "/chat",
            headers=self.headers,
            json={"message": "   "}
        )
        unauthenticated = self.client.post(
            "/chat",
            json={"message": "Question"}
        )

        self.assertEqual(missing_message.status_code, 422)
        self.assertEqual(blank_message.status_code, 422)
        self.assertEqual(unauthenticated.status_code, 401)

    def test_chat_endpoint_sanitizes_model_timeouts(self):
        with patch.object(
            api.ai_service,
            "generate_answer",
            side_effect=TimeoutError("private provider detail")
        ):
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"message": "Question"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private provider detail", response.text)

    def test_upload_retrieval_citations_and_persisted_history(self):
        def embedding(*args):
            task_type = args[1]
            return [1.0, 0.0] if task_type == "RETRIEVAL_DOCUMENT" else [1.0, 0.0]

        with (
            patch.object(rag, "create_embedding", side_effect=embedding),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="Revenue was $42 [1]."
            ) as generate
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={"file": ("annual.txt", b"Annual revenue was $42.", "text/plain")}
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={"title": "Annual results"}
            )
            conversation_id = conversation.json()["id"]
            answer = self.client.post(
                f"/api/conversations/{conversation_id}/messages",
                headers=self.headers,
                json={"message": "What was the annual revenue?"}
            )

        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["citations"][0]["filename"], "annual.txt")
        prompt = json.loads(generate.call_args.args[0])
        self.assertIn("conversation_history", prompt)
        history = self.client.get(
            f"/api/conversations/{conversation_id}/messages",
            headers=self.headers
        )
        self.assertEqual(len(history.json()["messages"]), 2)
        self.assertEqual(history.json()["messages"][1]["citations"][0]["index"], 1)

    def test_message_combines_document_business_and_task_context(self):
        with (
            patch.object(
                rag,
                "create_embedding",
                return_value=[1.0, 0.0]
            ),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="The open task concerns a $42 annual revenue report [1]."
            ) as generate,
            patch.object(
                api.chat_api,
                "search_knowledge",
                return_value=[{
                    "title": "Example Books",
                    "content": "Example Books is a small bookkeeping business.",
                    "source": "Prepared business knowledge"
                }]
            ),
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": (
                        "annual.txt",
                        b"Annual revenue was $42.",
                        "text/plain"
                    )
                }
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={"title": "Annual report task"}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={
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
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"][0]["filename"], "annual.txt")
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(
            prompt["retrieved_business_context"][0]["text"],
            "Example Books is a small bookkeeping business."
        )
        self.assertEqual(prompt["business_context"]["name"], "Example Books")
        self.assertEqual(
            prompt["task_context"]["current_task"],
            "Prepare annual report"
        )
        self.assertEqual(prompt["task_context"]["status"], "in progress")
        self.assertEqual(
            prompt["retrieved_document_context"][0]["text"],
            "Annual revenue was $42."
        )

    def test_missing_context_and_documents_are_kept_empty(self):
        with (
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="That information is not available in the supplied sources."
            ) as generate
        ):
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "What is my task status?"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["business_context"], {})
        self.assertEqual(prompt["task_context"], {})
        self.assertEqual(prompt["retrieved_document_context"], [])
        self.assertIn("ask a specific clarification question", prompt["instructions"])

    def test_business_question_uses_retrieved_prepared_context(self):
        with (
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(api.chat_api, "search_knowledge", return_value=[{
                "id": 5,
                "title": "HisabDo",
                "content": "HisabDo is an expense-management and digital-ledger product.",
                "source": "https://www.xicteksystems.com/",
                "metadata": {
                    "id": 5,
                    "source": "https://www.xicteksystems.com/",
                    "is_public": True
                },
                "score": 0.9
            }]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="HisabDo is an expense-management product."
            ) as generate
        ):
            conversation = self.client.post(
                "/api/conversations", headers=self.headers, json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "What is HisabDo?"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(len(prompt["retrieved_business_context"]), 1)
        self.assertEqual(prompt["retrieved_business_context"][0]["id"], 5)
        self.assertEqual(prompt["retrieved_business_context"][0]["score"], 0.9)
        self.assertEqual(
            prompt["retrieved_business_context"][0]["metadata"]["is_public"],
            True
        )
        self.assertEqual(prompt["retrieved_document_context"], [])
        self.assertEqual(response.json()["citations"], [])

    def test_prepared_business_retrieval_reaches_rag_prompt_end_to_end(self):
        def embedding(text, task_type):
            if task_type == "RETRIEVAL_QUERY" or text.startswith("HisabDo is"):
                return [1.0, 0.0]
            return [0.0, 1.0]

        self.business_search.stop()
        api.chat_api.vector_store.clear()
        try:
            with (
                patch.object(
                    api.chat_api,
                    "create_embedding",
                    side_effect=embedding
                ),
                patch.object(rag, "search_chunks", return_value=[]),
                patch.object(
                    api.ai_service,
                    "generate_answer",
                    return_value="HisabDo is a digital-ledger product."
                ) as generate
            ):
                conversation = self.client.post(
                    "/api/conversations", headers=self.headers, json={}
                ).json()
                response = self.client.post(
                    f"/api/conversations/{conversation['id']}/messages",
                    headers=self.headers,
                    json={"message": "What is HisabDo?"}
                )
        finally:
            api.chat_api.vector_store.clear()
            self.business_search.start()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(len(prompt["retrieved_business_context"]), 1)
        result = prompt["retrieved_business_context"][0]
        self.assertEqual(result["id"], 5)
        self.assertEqual(result["title"], "HisabDo")
        self.assertEqual(result["metadata"]["is_public"], True)
        self.assertGreaterEqual(
            result["score"],
            api.chat_api.MIN_RELEVANCE_SCORE
        )

    def test_customer_identifier_retrieves_and_cites_matching_record(self):
        with (
            patch.object(rag, "create_embedding", return_value=[1.0, 0.0]),
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="Maya Khan has an overdue balance of $120 [1]."
            ) as generate
        ):
            target_upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": (
                        "customer-104.txt",
                        b"Customer ID C-104. Name: Maya Khan. Balance: $120. Payment status: overdue.",
                        "text/plain"
                    )
                }
            )
            unrelated_upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": (
                        "customer-1045.txt",
                        b"Customer ID C-1045. Name: John Doe. Balance: $30.",
                        "text/plain"
                    )
                }
            )
            self.assertEqual(target_upload.status_code, 201, target_upload.text)
            self.assertEqual(unrelated_upload.status_code, 201, unrelated_upload.text)
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={
                    "message": "What is this customer's payment status?",
                    "customer_identifier": "C-104",
                    "business_context": {"name": "Example Books"},
                    "task_context": {"current_task": "Review customer balances"}
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        citations = response.json()["citations"]
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["filename"], "customer-104.txt")
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["customer_identifier"], "C-104")
        self.assertEqual(
            prompt["retrieved_customer_context"],
            [{
                "customer_identifier": "C-104",
                "citation": 1,
                "document": "customer-104.txt",
                "page": None
            }]
        )
        self.assertEqual(
            prompt["retrieved_document_context"][0]["text"],
            "Customer ID C-104. Name: Maya Khan. Balance: $120. Payment status: overdue."
        )
        self.assertEqual(prompt["business_context"]["name"], "Example Books")
        self.assertEqual(
            prompt["task_context"]["current_task"],
            "Review customer balances"
        )
        self.assertIn("Never invent customer names", prompt["instructions"])
        self.assertIn("If no record matches", prompt["instructions"])
        self.assertNotIn("John Doe", generate.call_args.args[0])

    def test_customer_context_is_user_scoped_when_identifier_is_missing(self):
        with (
            patch.object(rag, "create_embedding", return_value=[1.0, 0.0]),
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="That customer record is unavailable."
            ) as generate
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": (
                        "customer.txt",
                        b"Customer ID C-104. Name: Maya Khan.",
                        "text/plain"
                    )
                }
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            other_user = self.client.post("/api/auth/register", json={
                "email": "customer-reader@example.com",
                "password": "another-secure-password"
            }).json()
            other_headers = {
                "Authorization": "Bearer " + other_user["access_token"]
            }
            conversation = self.client.post(
                "/api/conversations",
                headers=other_headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=other_headers,
                json={
                    "message": "What is this customer's name?",
                    "customer_identifier": "C-104"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["customer_identifier"], "C-104")
        self.assertEqual(prompt["retrieved_customer_context"], [])
        self.assertEqual(prompt["retrieved_document_context"], [])
        self.assertNotIn("Maya Khan", generate.call_args.args[0])

    def test_irrelevant_retrieved_document_is_not_added_to_prompt(self):
        with (
            patch.object(
                rag,
                "create_embedding",
                side_effect=[[0.0, 1.0], [1.0, 0.0]]
            ),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="That information is not available in the sources."
            ) as generate
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={"file": ("unrelated.txt", b"Gardening tips.", "text/plain")}
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "What was the annual revenue?"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["retrieved_document_context"], [])

    def test_business_context_and_task_context_are_independently_optional(self):
        with (
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="I will use the supplied context."
            ) as generate
        ):
            requests = [
                {
                    "message": "What does this business do?",
                    "business_context": {"name": "Example Books"}
                },
                {
                    "message": "What is my current task?",
                    "task_context": {"current_task": "Prepare the annual report"}
                }
            ]
            for request in requests:
                conversation = self.client.post(
                    "/api/conversations",
                    headers=self.headers,
                    json={}
                ).json()
                response = self.client.post(
                    f"/api/conversations/{conversation['id']}/messages",
                    headers=self.headers,
                    json=request
                )
                self.assertEqual(response.status_code, 200, response.text)

        prompts = [
            json.loads(call.args[0])
            for call in generate.call_args_list
        ]
        self.assertEqual(prompts[0]["business_context"]["name"], "Example Books")
        self.assertEqual(prompts[0]["task_context"], {})
        self.assertEqual(
            prompts[1]["task_context"]["current_task"],
            "Prepare the annual report"
        )
        self.assertEqual(prompts[1]["business_context"], {})

    def test_prompt_uses_current_task_context_over_stale_history(self):
        prompt = json.loads(api.build_chat_prompt(
            "Task status",
            [{"role": "assistant", "content": "The task is still open."}],
            [],
            [],
            "What is the task status now?",
            None,
            api.TaskContext(status="complete")
        ))

        self.assertEqual(prompt["task_context"]["status"], "complete")
        self.assertEqual(len(prompt["conversation_history"]), 1)
        self.assertIn("prefer current context", prompt["instructions"])
        self.assertIn("not independently verified facts", prompt["instructions"])

    def test_prompt_deduplicates_sources_and_handles_malformed_business_hits(self):
        prompt = json.loads(api.build_chat_prompt(
            "Annual report",
            [],
            [
                {"title": "Prepared", "content": "Annual revenue was $42.", "source": "KB"},
                {"title": "Malformed", "content": None},
                {"title": "Other", "content": "Example Books is a bookkeeping business."}
            ],
            [{"citation": 1, "text": "Annual  revenue was $42."}],
            "What was annual revenue?",
            None,
            None
        ))

        self.assertEqual(len(prompt["retrieved_business_context"]), 1)
        self.assertEqual(
            prompt["retrieved_business_context"][0]["text"],
            "Example Books is a bookkeeping business."
        )
        self.assertIn("identify the conflict", prompt["instructions"])
        self.assertIn("Never disclose confidential", prompt["instructions"])

    def test_business_retrieval_failure_does_not_block_document_chat(self):
        with (
            patch.object(rag, "search_chunks", return_value=[{
                "document_id": "doc-1",
                "filename": "annual.txt",
                "page_number": None,
                "score": 0.9,
                "content": "Annual revenue was $42."
            }]),
            patch.object(
                api.chat_api,
                "search_knowledge",
                side_effect=RuntimeError("provider unavailable")
            ),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="Annual revenue was $42 [1]."
            ) as generate
        ):
            conversation = self.client.post(
                "/api/conversations", headers=self.headers, json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "What was annual revenue?"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["retrieved_business_context"], [])
        self.assertEqual(
            prompt["retrieved_document_context"][0]["text"],
            "Annual revenue was $42."
        )

    def test_business_context_is_not_retrieved_without_authentication(self):
        with patch.object(api.chat_api, "search_knowledge") as search:
            response = self.client.post(
                "/api/conversations/not-owned/messages",
                json={"message": "What does the business do?"}
            )

        self.assertEqual(response.status_code, 401)
        search.assert_not_called()

    def test_malformed_stored_embedding_does_not_fail_chat(self):
        with (
            patch.object(rag, "create_embedding", return_value=[1.0, 0.0]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="I could not find a matching source."
            )
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={"file": ("broken.txt", b"Some text.", "text/plain")}
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            with api.database.connect() as connection:
                connection.execute(
                    "UPDATE chunks SET embedding_json = ? WHERE document_id = ?",
                    ("not-json", upload.json()["id"])
                )
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "What does the document say?"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])

    def test_vector_index_upsert_search_metadata_and_delete(self):
        with patch.object(
            rag,
            "create_embedding",
            return_value=[1.0, 0.0]
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": (
                        "annual.txt",
                        b"Annual revenue was $42.",
                        "text/plain"
                    )
                }
            )
        self.assertEqual(upload.status_code, 201, upload.text)
        document_id = upload.json()["id"]
        api_status = self.client.get(
            "/api/vector-index",
            headers=self.headers
        )
        self.assertEqual(api_status.status_code, 200, api_status.text)
        self.assertEqual(api_status.json()["record_count"], 1)

        with api.database.connect() as connection:
            record = rag.get_embedding(connection, document_id, 0, 1)
            self.assertEqual(record["id"], f"{document_id}:0")
            self.assertEqual(record["content"], "Annual revenue was $42.")
            self.assertEqual(record["metadata"]["source"], "annual.txt")
            self.assertEqual(rag.get_index_status(connection, 1)["record_count"], 1)

            api.database.initialize_database()
            status = rag.get_index_status(connection, 1)
            self.assertEqual(status["embedding_dimensions"], 2)
            self.assertEqual(status["similarity_metric"], "cosine")

            stable_id = rag.update_embedding(
                connection,
                document_id,
                0,
                "Updated annual revenue was $42.",
                [0.9, 0.1],
                1
            )
            updated = rag.get_embedding(connection, document_id, 0, 1)
            self.assertEqual(stable_id, updated["id"])
            self.assertEqual(updated["content"], "Updated annual revenue was $42.")

            with patch.object(
                rag,
                "create_embedding",
                return_value=[1.0, 0.0]
            ):
                results = rag.search_chunks(
                    connection,
                    1,
                    "annual revenue",
                    limit=2,
                    filters={"filename": "annual.txt"}
                )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], stable_id)
            self.assertEqual(results[0]["metadata"]["page_number"], 1)

            self.assertFalse(
                rag.delete_embedding(connection, document_id, 0, user_id=999)
            )
            self.assertTrue(
                rag.delete_embedding(connection, document_id, 0, user_id=1)
            )
            self.assertIsNone(rag.get_embedding(connection, document_id, 0, 1))
            self.assertEqual(rag.get_index_status(connection, 1)["record_count"], 0)

    def test_vector_index_rejects_invalid_and_mismatched_embeddings(self):
        with patch.object(
            rag,
            "create_embedding",
            return_value=[1.0, 0.0]
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={
                    "file": ("notes.txt", b"A valid chunk.", "text/plain")
                }
            )
        self.assertEqual(upload.status_code, 201, upload.text)

        with api.database.connect() as connection:
            with self.assertRaises(ValueError):
                rag.update_embedding(
                    connection,
                    upload.json()["id"],
                    0,
                    "Invalid dimension update.",
                    [1.0, 0.0, 0.0]
                )
            with self.assertRaises(ValueError):
                rag.serialize_embedding([float("nan")])
            with self.assertRaises(ValueError):
                rag.serialize_embedding([0.0, 0.0])

    def test_invalid_nested_context_is_rejected(self):
        conversation = self.client.post(
            "/api/conversations",
            headers=self.headers,
            json={}
        ).json()
        response = self.client.post(
            f"/api/conversations/{conversation['id']}/messages",
            headers=self.headers,
            json={
                "message": "What is the task status?",
                "task_context": {"status": ["not", "a", "status"]}
            }
        )

        self.assertEqual(response.status_code, 422)

    def test_retrieval_never_includes_another_users_documents(self):
        with (
            patch.object(rag, "create_embedding", return_value=[1.0, 0.0]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value="No matching document was found."
            ) as generate
        ):
            upload = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={"file": ("private.txt", b"Private business detail.", "text/plain")}
            )
            self.assertEqual(upload.status_code, 201, upload.text)
            second = self.client.post("/api/auth/register", json={
                "email": "other@example.com",
                "password": "another-secure-password"
            }).json()
            other_headers = {
                "Authorization": "Bearer " + second["access_token"]
            }
            conversation = self.client.post(
                "/api/conversations",
                headers=other_headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=other_headers,
                json={
                    "message": "What does this business earn?",
                    "business_context": {
                        "business_id": "another-business",
                        "name": "Another business"
                    }
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"], [])
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["retrieved_document_context"], [])
        self.assertEqual(prompt["business_context"]["name"], "Another business")
        self.assertNotIn("Private business detail.", generate.call_args.args[0])

    def test_llm_failure_returns_service_error_without_saving_message(self):
        with (
            patch.object(rag, "search_chunks", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                side_effect=RuntimeError("private provider detail")
            )
        ):
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "Can you help?"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private provider", response.text)
        messages = self.client.get(
            f"/api/conversations/{conversation['id']}/messages",
            headers=self.headers
        )
        self.assertEqual(messages.json()["messages"], [])

    def test_retrieval_failure_returns_service_error(self):
        with patch.object(
            rag,
            "search_chunks",
            side_effect=RuntimeError("private vector-store detail")
        ):
            conversation = self.client.post(
                "/api/conversations",
                headers=self.headers,
                json={}
            ).json()
            response = self.client.post(
                f"/api/conversations/{conversation['id']}/messages",
                headers=self.headers,
                json={"message": "Can you search the documents?"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private vector-store", response.text)

    def test_user_cannot_read_or_delete_another_users_conversation(self):
        conversation = self.client.post(
            "/api/conversations",
            headers=self.headers,
            json={"title": "Private"}
        ).json()
        second = self.client.post("/api/auth/register", json={
            "email": "other@example.com",
            "password": "another-secure-password"
        }).json()
        other_headers = {"Authorization": f"Bearer {second['access_token']}"}
        response = self.client.get(
            f"/api/conversations/{conversation['id']}/messages",
            headers=other_headers
        )
        self.assertEqual(response.status_code, 404)

    def test_upload_rejects_unsupported_files(self):
        result = self.client.post(
            "/api/documents",
            headers=self.headers,
            files={"file": ("notes.csv", b"a,b", "text/csv")}
        )
        self.assertEqual(result.status_code, 415)

    def test_embedding_provider_value_error_returns_service_error(self):
        with patch.object(
            rag,
            "create_embedding",
            side_effect=ValueError("private provider detail")
        ):
            response = self.client.post(
                "/api/documents",
                headers=self.headers,
                files={"file": ("notes.txt", b"Business note.", "text/plain")}
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private provider", response.text)

    def test_authentication_rate_limit_is_enforced(self):
        with api.database.connect() as connection:
            connection.execute("DELETE FROM rate_limit_buckets")
        payload = {"email": "missing@example.com", "password": "not-a-password"}
        responses = [
            self.client.post("/api/auth/login", json=payload)
            for _ in range(11)
        ]
        self.assertTrue(all(response.status_code == 401 for response in responses[:10]))
        self.assertEqual(responses[10].status_code, 429)
        self.assertIn("Retry-After", responses[10].headers)


if __name__ == "__main__":
    unittest.main()
