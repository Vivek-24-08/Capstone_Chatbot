"""Exercise real LangChain request preparation against a fake network boundary."""
from types import SimpleNamespace
from google.ai.generativelanguage_v1beta.types import GenerateContentResponse, Candidate, Content, Part
from langchain_core.messages import HumanMessage
from config.settings import settings
from rag_pipeline.llm_service import get_chat_model


def response():
    return GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text="OK")]))])


def test_normal_chat_passes_deadline_to_transport(monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return response()
    model = get_chat_model()
    monkeypatch.setattr(model, "client", SimpleNamespace(generate_content=generate))
    assert model.invoke([HumanMessage(content="Reply OK")]).content == "OK"
    assert calls[0]["timeout"] == settings.request_timeout_seconds
    assert calls[0]["retry"] is None


def test_stream_passes_deadline_to_transport(monkeypatch):
    calls = []
    def stream(**kwargs):
        calls.append(kwargs)
        return iter([response()])
    model = get_chat_model()
    monkeypatch.setattr(model, "client", SimpleNamespace(stream_generate_content=stream))
    assert "".join(chunk.content for chunk in model.stream([HumanMessage(content="Reply OK")])) == "OK"
    assert calls[0]["timeout"] == settings.request_timeout_seconds
    assert calls[0]["retry"] is None
