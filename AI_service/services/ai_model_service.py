import os

from google import genai
from google.genai import types

from AI_service import settings
from AI_service.context.business_context import SYSTEM_PROMPT


MODEL = settings.GEMINI_MODEL
client = genai.Client(
    http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT_MS)
)


def generate_answer(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text


def create_embedding(text: str, task_type: str) -> list[float]:
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    result = client.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise RuntimeError("The embedding provider returned no vector.")
    return [float(value) for value in result.embeddings[0].values]
