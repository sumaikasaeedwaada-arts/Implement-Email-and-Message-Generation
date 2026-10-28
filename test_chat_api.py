import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from AI_service import chat_api
from AI_service import app as service_app
from AI_service import ai_service


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(chat_api.app)
        self.original_knowledge_base = list(chat_api.knowledge_base)
        chat_api.conversations.clear()
        chat_api.vector_store.clear()
        self.request_data = {
            "conversation_id": "test-conversation",
            "message": "What does HisabDo do?",
            "user": "",
            "business": "",
            "customer": "",
            "task": "",
            "notes": ""
        }

    def tearDown(self):
        chat_api.knowledge_base[:] = self.original_knowledge_base
        chat_api.vector_store.clear()

    def test_chat_uses_retrieved_business_context_and_previous_history(self):
        chat_api.conversations["test-conversation"] = [
            {"role": "user", "message": "Earlier question", "time": "now"},
            {"role": "assistant", "message": "Earlier answer", "time": "now"}
        ]
        model_response = MagicMock(text="HisabDo is an expense-management app.")
        with (
            patch.object(chat_api, "search_knowledge", return_value=[{
                "id": 5,
                "title": "HisabDo",
                "content": "HisabDo is an expense-management and digital-ledger product.",
                "source": "https://www.xicteksystems.com/",
                "score": 0.9
            }]),
            patch.object(
                chat_api.client.models,
                "generate_content",
                return_value=model_response
            ) as generate
        ):
            response = self.client.post("/chat", json=self.request_data)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["response"], model_response.text)
        call = generate.call_args.kwargs
        prompt = json.loads(call["contents"])
        self.assertEqual(
            prompt["retrieved_knowledge"][0]["title"],
            "HisabDo"
        )
        self.assertEqual(len(prompt["conversation_history"]), 2)
        self.assertEqual(prompt["current_question"], self.request_data["message"])
        self.assertIn("Do not invent", call["config"].system_instruction)

        saved = chat_api.conversations["test-conversation"]
        self.assertEqual(len(saved), 4)
        self.assertEqual(saved[-2]["message"], self.request_data["message"])

    def test_unknown_knowledge_is_not_presented_as_retrieved_fact(self):
        with (
            patch.object(chat_api, "search_knowledge", return_value=[]),
            patch.object(
                chat_api.client.models,
                "generate_content",
                return_value=MagicMock(text="That information is not available.")
            ) as generate
        ):
            response = self.client.post("/chat", json=self.request_data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["retrieved_knowledge"], [])
        self.assertEqual(
            json.loads(generate.call_args.kwargs["contents"])["retrieved_knowledge"],
            []
        )

    def test_chat_errors_are_sanitized_and_not_saved(self):
        with (
            patch.object(chat_api, "search_knowledge", return_value=[]),
            patch.object(
                chat_api.client.models,
                "generate_content",
                side_effect=RuntimeError("private upstream credential detail")
            )
        ):
            response = self.client.post("/chat", json=self.request_data)

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("credential", response.text)
        self.assertNotIn("test-conversation", chat_api.conversations)

    def test_conversation_history_is_bounded(self):
        chat_api.conversations["test-conversation"] = [
            {"role": "user", "message": str(index), "time": "now"}
            for index in range(chat_api.MAX_HISTORY_MESSAGES)
        ]
        with (
            patch.object(chat_api, "search_knowledge", return_value=[]),
            patch.object(
                chat_api.client.models,
                "generate_content",
                return_value=MagicMock(text="Answer")
            )
        ):
            response = self.client.post("/chat", json=self.request_data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            len(chat_api.conversations["test-conversation"]),
            chat_api.MAX_HISTORY_MESSAGES
        )

    def test_embedding_retrieval_discards_low_relevance_matches(self):
        chat_api.vector_store.clear()
        public_items = [
            item for item in chat_api.knowledge_base
            if item.get("is_public") is True
        ]
        vectors = [[0.0, 1.0]] * len(public_items)
        vectors[4] = [1.0, 0.0]
        vectors.append([1.0, 0.0])
        with patch.object(chat_api, "create_embedding", side_effect=vectors):
            results = chat_api.search_knowledge("HisabDo")

        self.assertEqual([item["title"] for item in results], ["HisabDo"])
        self.assertGreaterEqual(results[0]["score"], chat_api.MIN_RELEVANCE_SCORE)

    def test_knowledge_writes_require_configured_api_key_and_clear_embeddings(self):
        chat_api.vector_store.append({"id": 1, "embedding": [1.0]})
        with patch.dict(os.environ, {"XICTEK_KNOWLEDGE_API_KEY": ""}):
            disabled = self.client.post(
                "/knowledge",
                json={"title": "Test", "content": "Test content"}
            )
        self.assertEqual(disabled.status_code, 503)

        with patch.dict(os.environ, {"XICTEK_KNOWLEDGE_API_KEY": "test-key"}):
            read_unauthorized = self.client.get("/knowledge")
            unauthorized = self.client.post(
                "/knowledge",
                json={"title": "Test", "content": "Test content"},
                headers={"X-API-Key": "wrong"}
            )
            authorized = self.client.post(
                "/knowledge",
                json={"title": "Test", "content": "Test content"},
                headers={"X-API-Key": "test-key"}
            )

        self.assertEqual(read_unauthorized.status_code, 401)
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(chat_api.vector_store, [])
        self.assertEqual(
            authorized.json()["knowledge"]["source"],
            "Admin-provided knowledge"
        )
        self.assertFalse(authorized.json()["knowledge"]["is_public"])
        with patch.dict(os.environ, {"XICTEK_KNOWLEDGE_API_KEY": "test-key"}):
            published = self.client.post(
                "/knowledge",
                json={
                    "title": "Public service detail",
                    "content": "Public service detail.",
                    "is_public": True
                },
                headers={"X-API-Key": "test-key"}
            )
        self.assertTrue(published.json()["knowledge"]["is_public"])

    def test_private_admin_knowledge_is_not_retrieved_by_public_chat(self):
        with (
            patch.object(chat_api, "knowledge_base", [{
                "id": 1,
                "title": "Internal pricing",
                "content": "Confidential negotiated pricing is $10 per unit.",
                "source": "Admin-provided knowledge",
                "is_public": False
            }]),
            patch.object(chat_api, "create_embedding", return_value=[1.0, 0.0]),
            patch.object(
                chat_api.client.models,
                "generate_content",
                return_value=MagicMock(text="That information is unavailable.")
            ) as generate
        ):
            response = self.client.post("/chat", json=self.request_data)

        self.assertEqual(response.status_code, 200)
        prompt = json.loads(generate.call_args.kwargs["contents"])
        self.assertEqual(prompt["retrieved_knowledge"], [])
        self.assertNotIn("Confidential negotiated pricing", generate.call_args.kwargs["contents"])

    def test_empty_chat_message_is_rejected(self):
        request_data = dict(self.request_data, message="")
        response = self.client.post("/chat", json=request_data)
        self.assertEqual(response.status_code, 422)

    def test_ask_ai_includes_verified_business_context(self):
        with patch.object(
            ai_service.client.models,
            "generate_content",
            return_value=MagicMock(text="Answer")
        ) as generate:
            result = ai_service.ask_ai("What is HisabDo?")

        self.assertEqual(result, "Answer")
        call = generate.call_args.kwargs
        self.assertIn("HisabDo", call["contents"])
        self.assertIn("Do not invent", call["config"].system_instruction)

    def test_ask_endpoint_returns_sanitized_error(self):
        client = TestClient(service_app.app)
        with patch.object(
            service_app,
            "ask_ai",
            side_effect=RuntimeError("private upstream detail")
        ):
            response = client.get("/ask", params={"question": "Known question"})

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private upstream", response.text)

    def test_root_chatbot_module_is_safe_to_import(self):
        import chatbot

        self.assertTrue(callable(chatbot.run_cli))


if __name__ == "__main__":
    unittest.main()
