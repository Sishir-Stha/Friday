from friday.llm.prompts import FRIDAY_SYSTEM_PROMPT


def test_friday_system_prompt_preserves_core_response_contract() -> None:
    assert "You are Friday, a private local AI assistant" in FRIDAY_SYSTEM_PROMPT
    assert "operating in local AI mode" in FRIDAY_SYSTEM_PROMPT
    assert "Return only the final answer" in FRIDAY_SYSTEM_PROMPT
    assert "Do not produce or expose internal reasoning" in FRIDAY_SYSTEM_PROMPT
    assert "exact response formats as strict output constraints" in (
        FRIDAY_SYSTEM_PROMPT
    )
    assert "output only that value, copied exactly" in FRIDAY_SYSTEM_PROMPT
