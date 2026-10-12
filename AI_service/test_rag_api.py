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
