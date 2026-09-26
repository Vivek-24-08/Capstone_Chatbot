"""Optional setup checks: python -m scripts.doctor [--probe]."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def run_checks(probe=False):
    from config.settings import settings, PROJECT_ROOT
    from utils.service_errors import classify_error
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"check": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            issue = classify_error(exc)
            checks.append({"check": name, "ok": False, "code": issue.code,
                           "detail": issue.message})

    try:
        settings.validate()
        checks.append({"check": "Configuration", "ok": True, "detail": "Required settings are present (credentials are never printed)."})
    except ValueError as exc:
        checks.append({"check": "Configuration", "ok": False, "detail": str(exc)})
        return checks
    checks.append({"check": "Environment", "ok": True, "detail": f"Python {sys.version.split()[0]}; project {PROJECT_ROOT}"})
    check("Documents", lambda: _document_summary(Path(settings.pdf_data_dir)))
    from vector_store import chroma_manager
    check("Vector index", lambda: f"{chroma_manager.count()} indexed chunks")
    check("Index configuration", chroma_manager.assert_compatible_index)
    for package in ("streamlit", "chromadb", "langchain-openai", "sentence-transformers"):
        check(f"Package {package}", lambda p=package: importlib.metadata.version(p))
    if probe:
        from embeddings.embedding_service import generate_query_embedding
        from rag_pipeline.llm_service import get_chat_model, acquire_chat_slot
        from rag_pipeline.rag_pipeline import _response_text
        from langchain_core.messages import HumanMessage
        check("Embedding connection", lambda: f"Returned {len(generate_query_embedding('Connection test'))} dimensions")

        def chat():
            acquire_chat_slot()
            response = get_chat_model().invoke([HumanMessage(content="Reply with OK only.")])
            if not _response_text(response.content).strip():
                raise ValueError("The model returned an empty response.")
            return "Chat model returned text."
        check("Chat connection", chat)
    return checks


def _document_summary(folder):
    if not folder.is_dir():
        raise FileNotFoundError("PDF directory missing")
    return f"{len(list(folder.glob('*.pdf')))} PDFs in {folder}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Make a small embedding and chat request (provider usage may apply).")
    args = parser.parse_args()
    checks = run_checks(args.probe)
    print(json.dumps(checks, indent=2))
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
