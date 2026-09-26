# ==============================================================================
# tests/test_provider_switching.py
# ------------------------------------------------------------------------------
# Tests that config/settings.py's validate() correctly enforces (or does NOT
# enforce) credentials depending on LLM_PROVIDER / EMBEDDING_PROVIDER --
# covering the "Gemini can be fully replaced by keyless Databricks" claim.
#
# These are pure config/logic tests -- no real Databricks workspace or
# Gemini key is used or needed; we never actually construct a chat model or
# call an endpoint here, only exercise the validation rules.
# ==============================================================================

import pytest

from config.settings import settings


def test_validate_requires_gemini_key_when_llm_provider_is_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "gemini_api_key", "")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate()


def test_validate_requires_openrouter_key_and_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        settings.validate()

    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "openrouter_model", "")
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        settings.validate()


def test_get_chat_model_dispatches_to_openrouter(monkeypatch):
    import rag_pipeline.llm_service as llm_service
    import langchain_openai

    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key-not-real")
    monkeypatch.setattr(settings, "openrouter_model", "openai/gpt-4o-mini")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    class ChatOpenAI:
        def __init__(self, **kwargs):
            self.options = kwargs
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", ChatOpenAI)
    model = llm_service.get_chat_model()
    assert model.options["model"] == "openai/gpt-4o-mini"
    assert model.options["base_url"] == "https://openrouter.ai/api/v1"


def test_validate_passes_fully_keyless_when_both_providers_are_databricks(monkeypatch):
    # This is the "fully replace Gemini" configuration: no Gemini key
    # anywhere, both chat and embeddings routed to Databricks.
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "databricks")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "databricks_host", "")
    monkeypatch.setattr(settings, "databricks_token", "")

    # Should not raise -- Databricks auth is resolved at call time by the
    # SDK's own ambient-credential chain, never validated as a config value.
    settings.validate()


def test_validate_rejects_unknown_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        settings.validate()


def test_validate_rejects_unknown_embedding_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(settings, "embedding_provider", "azure_openai")

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        settings.validate()


def test_get_chat_model_dispatches_to_gemini(monkeypatch):
    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    model = llm_service.get_chat_model()

    from langchain_google_genai import ChatGoogleGenerativeAI
    assert isinstance(model, ChatGoogleGenerativeAI)


def test_get_chat_model_dispatches_to_databricks(monkeypatch):
    import rag_pipeline.llm_service as llm_service
    import langchain_databricks

    monkeypatch.setattr(settings, "llm_provider", "databricks")
    monkeypatch.setattr(llm_service, "_llm_instance", None)

    # This is a dispatch test, not a live workspace authentication test.
    class ChatDatabricks:
        def __init__(self, **kwargs):
            self.options = kwargs
    monkeypatch.setattr(langchain_databricks, "ChatDatabricks", ChatDatabricks)

    model = llm_service.get_chat_model()

    assert type(model).__name__ == "ChatDatabricks"
    assert model.options["endpoint"] == settings.databricks_llm_endpoint
