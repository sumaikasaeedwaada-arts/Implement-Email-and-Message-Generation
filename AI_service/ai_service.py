import time

from AI_service.context.business_context import BUSINESS_CONTEXT, SYSTEM_PROMPT
from AI_service.services.ai_model_service import (
    MODEL,
    client,
    create_embedding,
    generate_answer,
)


def ask_ai(question):
    prompt = (
        f"VERIFIED BUSINESS CONTEXT:\n{BUSINESS_CONTEXT}\n\n"
        f"USER QUESTION:\n{question}"
    )
    return generate_answer(prompt)


def run_cli():
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )

    print("=" * 45)
    print("🤖 AI CHATBOT")
    print("=" * 45)
    print("Type 'exit' to stop.")
    print("Type 'clear' to start a new conversation.")
    print()

    while True:
        message = input("You: ").strip()

        if not message:
            print("Bot: Please enter a message.")
            print()
            continue

        if message.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        if message.lower() == "clear":
            chat = client.chats.create(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
            print("Bot: Chat cleared! 🧹")
            print()
            continue

        print("Bot: Thinking... ⏳")

        for attempt in range(3):
            try:
                response = chat.send_message(message)
                print("Bot:", response.text)
                print()
                break
            except Exception as exc:
                if "503" in str(exc) and attempt < 2:
                    print("Bot: Server busy. Retrying... 🔄")
                    time.sleep(3)
                else:
                    print("Bot: Sorry, something went wrong. ❌")
                    print()
                    break


if __name__ == "__main__":
    run_cli()
