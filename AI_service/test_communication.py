import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from AI_service import api


class CommunicationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "DATABASE_PATH": os.path.join(
                self.temp_directory.name,
                "communication.sqlite"
            ),
            "AUTH_SECRET": "test-auth-secret-not-for-production"
        })
        self.env.start()
        api.database.initialize_database()
        self.client = TestClient(api.app)
        self.client.__enter__()
        registration = self.client.post("/api/auth/register", json={
            "email": "communication-owner@example.com",
            "password": "a-secure-test-password"
        })
        self.assertEqual(registration.status_code, 201)
        self.user_id = registration.json()["user"]["id"]
        self.headers = {
            "Authorization": f"Bearer {registration.json()['access_token']}"
        }

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env.stop()
        self.temp_directory.cleanup()

    def add_record(self, document_id: str, user_id: int, content: str):
        with api.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, user_id, filename, media_type, status, created_at)
                VALUES (?, ?, ?, 'text/plain', 'ready', '2026-10-25T12:00:00+00:00')
                """,
                (document_id, user_id, f"{document_id}.txt")
            )
            connection.execute(
                """
                INSERT INTO chunks
                    (document_id, chunk_index, page_number, content, embedding_json)
                VALUES (?, 0, 1, ?, '[]')
                """,
                (document_id, content)
            )

    def test_email_generation_returns_subject_and_formatted_body(self):
        response_json = json.dumps({
            "subject": "Following Up on Our Meeting",
            "body": "Hello John,\n\nI am following up on our meeting.\n\nBest regards,\nAri"
        })
        with (
            patch.object(
                api.chat_api,
                "search_knowledge",
                return_value=[{
                    "title": "HisabDo",
                    "source": "public website",
                    "content": "HisabDo is an expense-management product.",
                    "is_public": True
                }]
            ) as search,
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=response_json
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={
                    "communication_type": "email",
                    "purpose": "follow_up",
                    "tone": "professional",
                    "length": "medium",
                    "language": "English",
                    "recipient_name": "John",
                    "recipient_context": "Ask about HisabDo expense tracking.",
                    "sender_name": "Ari",
                    "additional_instructions": (
                        "Follow up about the previous meeting."
                    )
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["content"]
        self.assertEqual(content["subject"], "Following Up on Our Meeting")
        self.assertIn("Hello John", content["body"])
        self.assertIn("Best regards,\nAri", content["body"])
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(
            prompt["verified_public_business_context"][0]["title"],
            "HisabDo"
        )
        self.assertEqual(
            prompt["generation_options"]["length_guidance"],
            "about 80-140 words in the body"
        )
        self.assertIn("HisabDo", search.call_args.args[0])
        self.assertIn("Never invent", prompt["instructions"])

    def test_short_message_has_no_email_fields_or_formatting(self):
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "message": "Hi John, your appointment is confirmed for [date]."
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={
                    "communication_type": "message",
                    "purpose": "invitation",
                    "tone": "friendly",
                    "length": "short",
                    "recipient_name": "John"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["content"],
            {"message": "Hi John, your appointment is confirmed for [date]."}
        )
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["generation_options"]["tone"], "friendly")
        self.assertEqual(prompt["generation_options"]["length"], "short")
        self.assertEqual(prompt["generation_options"]["purpose"], "invitation")
        self.assertIn("do not add an email subject", prompt["instructions"])
        self.assertIn("bracketed placeholder", prompt["instructions"])

    def test_default_options_and_missing_recipient_use_no_invented_identity(self):
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "subject": "A Quick Question",
                    "body": "Hello [recipient name], could you share [missing detail]?"
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={"communication_type": "email"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(
            prompt["generation_options"]["purpose"],
            "customer support"
        )
        self.assertEqual(prompt["generation_options"]["tone"], "professional")
        self.assertEqual(prompt["generation_options"]["length"], "medium")
        self.assertIsNone(prompt["recipient"]["name"])
        self.assertEqual(
            prompt["authorized_customer_record_excerpts"],
            []
        )
        self.assertIn("[recipient name]", response.json()["content"]["body"])
        self.assertNotIn("signature", response.json()["content"]["body"].lower())

    def test_custom_purpose_and_language_are_passed_to_the_model(self):
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "message": "Gracias por su paciencia; le responderemos pronto."
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={
                    "communication_type": "message",
                    "purpose": "custom",
                    "custom_purpose": "Thank a customer for their patience",
                    "tone": "empathetic",
                    "length": "long",
                    "language": "Spanish"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(
            prompt["generation_options"]["purpose"],
            "Thank a customer for their patience"
        )
        self.assertEqual(prompt["generation_options"]["tone"], "empathetic")
        self.assertEqual(prompt["generation_options"]["length"], "long")
        self.assertEqual(prompt["generation_options"]["language"], "Spanish")

    def test_customer_context_is_requested_only_for_identifier_and_user_scoped(self):
        self.add_record(
            "owned-record",
            self.user_id,
            "Customer C-104 asked to reschedule the delivery to Friday."
        )
        second_user = self.client.post("/api/auth/register", json={
            "email": "another-owner@example.com",
            "password": "a-secure-test-password"
        }).json()["user"]["id"]
        self.add_record(
            "other-record",
            second_user,
            "Customer C-104 has a confidential balance of $900."
        )
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "subject": "Delivery Follow-Up",
                    "body": "Hello, your delivery request is noted."
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={
                    "communication_type": "email",
                    "purpose": "follow_up",
                    "customer_identifier": "C-104"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = generate.call_args.args[0]
        self.assertIn("reschedule the delivery to Friday", prompt)
        self.assertNotIn("confidential balance", prompt)

    def test_retrieved_business_information_must_be_public(self):
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[
                {
                    "title": "Private",
                    "content": "Do not disclose this.",
                    "is_public": False
                },
                {
                    "title": "Unmarked",
                    "content": "Visibility is unknown."
                }
            ]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "message": "Please contact us with your question."
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={"communication_type": "message"}
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["verified_public_business_context"], [])

    def test_validation_authentication_and_custom_purpose_requirements(self):
        missing_type = self.client.post(
            "/api/communications/generate",
            headers=self.headers,
            json={}
        )
        invalid_tone = self.client.post(
            "/api/communications/generate",
            headers=self.headers,
            json={"communication_type": "email", "tone": "casual"}
        )
        missing_custom_purpose = self.client.post(
            "/api/communications/generate",
            headers=self.headers,
            json={
                "communication_type": "email",
                "purpose": "custom"
            }
        )
        unauthenticated = self.client.post(
            "/api/communications/generate",
            json={"communication_type": "email"}
        )

        self.assertEqual(missing_type.status_code, 422)
        self.assertEqual(invalid_tone.status_code, 422)
        self.assertEqual(missing_custom_purpose.status_code, 422)
        self.assertEqual(unauthenticated.status_code, 401)

    def test_provider_timeout_and_invalid_output_return_sanitized_errors(self):
        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                side_effect=TimeoutError("private provider details")
            )
        ):
            timed_out = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={"communication_type": "message"}
            )

        with (
            patch.object(api.chat_api, "search_knowledge", return_value=[]),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value='{"subject": "missing body"}'
            )
        ):
            invalid = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={"communication_type": "email"}
            )

        self.assertEqual(timed_out.status_code, 503)
        self.assertNotIn("private provider", timed_out.text)
        self.assertEqual(invalid.status_code, 502)
        self.assertNotIn("missing body", invalid.text)

    def test_business_retrieval_failure_can_continue_without_retrieved_facts(self):
        with (
            patch.object(
                api.chat_api,
                "search_knowledge",
                side_effect=RuntimeError("embedding service detail")
            ),
            patch.object(
                api.ai_service,
                "generate_answer",
                return_value=json.dumps({
                    "message": "Please share the details you would like us to confirm."
                })
            ) as generate
        ):
            response = self.client.post(
                "/api/communications/generate",
                headers=self.headers,
                json={
                    "communication_type": "message",
                    "purpose": "announcement",
                    "tone": "formal",
                    "length": "long",
                    "language": "Urdu"
                }
            )

        self.assertEqual(response.status_code, 200, response.text)
        prompt = json.loads(generate.call_args.args[0])
        self.assertEqual(prompt["verified_public_business_context"], [])
        self.assertEqual(prompt["generation_options"]["language"], "Urdu")
        self.assertEqual(prompt["generation_options"]["length"], "long")
        self.assertEqual(prompt["generation_options"]["tone"], "formal")


if __name__ == "__main__":
    unittest.main()
