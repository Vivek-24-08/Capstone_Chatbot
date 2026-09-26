"""Exercise the real Streamlit script using its headless UI test runner."""
from pathlib import Path
from types import SimpleNamespace

from config.settings import settings
from rag_pipeline.llm_service import get_chat_model
from scripts.ingest import run_ingestion

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")


def test_missing_key_shows_setup_message_not_traceback(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("OPENROUTER_API_KEY" in item.value for item in app.error)


def test_ui_returns_streamed_answer_and_evidence(sample_pdf, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "pdf_data_dir", str(sample_pdf.parent))
    monkeypatch.setattr(settings, "enable_mlflow", False)
    run_ingestion()
    model = get_chat_model()
    monkeypatch.setattr(type(model), "stream", lambda *a, **k: iter([SimpleNamespace(content="Your deductible is $250.")]))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    app.chat_input[0].set_value("What is my annual deductible?").run()
    assert not app.exception
    answer = app.session_state["messages"][-1]
    assert answer["status"] == "ok"
    assert answer["content"] == "Your deductible is $250."
    assert answer["sources"]
    assert len(app.session_state["messages"]) == 2
    assert any("Your coverage, in conversation." in item.value for item in app.markdown)
    next(button for button in app.button if "New conversation" in button.label).click().run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert app.session_state["rag_pipeline"].memory.get_history() == []


def test_ui_keeps_evidence_when_chat_credentials_fail(sample_pdf, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "pdf_data_dir", str(sample_pdf.parent))
    monkeypatch.setattr(settings, "enable_mlflow", False)
    run_ingestion()
    model = get_chat_model()
    def fail(*args, **kwargs):
        raise RuntimeError("401 invalid authentication credentials")
    monkeypatch.setattr(type(model), "stream", fail)
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.chat_input[0].set_value("What is my annual deductible?").run()
    assert not app.exception
    assert any("credentials" in item.value for item in app.error)
    answer = app.session_state["messages"][-1]
    assert answer["status"] == "error"
    assert answer["sources"]
    assert app.session_state["rag_pipeline"].memory.get_history() == []
