import pytest

from friday.llm.ollama_client import OllamaClient

pytestmark = pytest.mark.integration


def test_ollama_available() -> None:
    client = OllamaClient()

    assert client.is_available()


def test_ollama_chat() -> None:
    client = OllamaClient()

    response = client.chat(
        [
            {
                "role": "user",
                "content": "Reply with exactly: Friday AI connected",
            }
        ]
    )

    assert response.content.strip() == "Friday AI connected"

    print("\n===== FRIDAY ANSWER =====")
    print(response.content)
