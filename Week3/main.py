
import os

from dotenv import load_dotenv

from llm_client import GeminiProvider, OllamaProvider, LLMClient
from usage_logger import UsageLogger

load_dotenv(override=True)


def main():
    usage_logger = None

    try:
        # Connect to MongoDB
        usage_logger = UsageLogger()
        print("Connected to MongoDB successfully!")

        # Select the LLM provider
        provider_name = input(
            "Choose provider (gemini/ollama): "
        ).strip().lower()

        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found in .env"
                )

            provider = GeminiProvider(api_key)

        elif provider_name == "ollama":
            provider = OllamaProvider("llama3.2")

        else:
            raise ValueError("Invalid provider selected")

        # Create LLM client
        client = LLMClient(provider, usage_logger)

        # Get user prompt
        prompt = input("Enter your question: ")

        # Generate response
        try:
            answer = client.generate(prompt)

            print("\nLLM Response:")
            print(answer)

        except Exception as e:
            print("\nGemini is temporarily unavailable.")
            print("Please try again later.")
            print("Error:", e)

    except Exception as e:
        print("\nApplication error:", e)

    finally:
        if usage_logger is not None:
            usage_logger.close()
            print("MongoDB connection closed.")


if __name__ == "__main__":
    main()
