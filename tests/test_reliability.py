"""Regression tests for sandbox failures, index publication and session isolation."""
from types import SimpleNamespace
import pytest

from config.settings import settings
from scripts import ingest
from vector_store import chroma_manager, metadata_table
from utils.rate_limiter import SlidingWindowRateLimiter


@pytest.fixture
def indexed(sample_pdf, monkeypatch):
    monkeypatch.setattr(settings, "pdf_data_dir", str(sample_pdf.parent))
    ingest.run_ingestion()
    return sample_pdf


def test_embedding_outage_preserves_published_index(indexed, monkeypatch):
    before = metadata_table.get_index_state()
    count = chroma_manager.count()
    def fail(texts):
        raise RuntimeError("embedding unavailable")
    monkeypatch.setattr(ingest, "generate_embeddings", fail)
    with pytest.raises(RuntimeError):
        ingest.run_ingestion(force=True)
    assert metadata_table.get_index_state() == before
    assert chroma_manager.count() == count
    assert metadata_table.query_by_document(indexed.name)


def test_publish_failure_preserves_old_index(indexed, monkeypatch):
    before = metadata_table.get_index_state()
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(metadata_table, "publish_generation", fail)
    with pytest.raises(OSError):
        ingest.run_ingestion(force=True)
    assert metadata_table.get_index_state() == before
    assert chroma_manager.count() > 0


def test_sqlite_publish_rolls_back_partial_writes(indexed):
    import sqlite3
    before = metadata_table.get_index_state()
    old_rows = metadata_table.query_by_document(indexed.name)
    with sqlite3.connect(settings.metadata_db_path) as connection:
        connection.execute("CREATE TRIGGER fail_publish BEFORE INSERT ON index_state BEGIN SELECT RAISE(ABORT, 'simulated commit failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ingest.run_ingestion(force=True)
    assert metadata_table.get_index_state() == before
    assert metadata_table.query_by_document(indexed.name) == old_rows
    assert chroma_manager.count() == len(old_rows)


def test_unchanged_documents_do_not_reembed(indexed, monkeypatch):
    def fail(texts):
        raise AssertionError("Unchanged documents should reuse their index")
    monkeypatch.setattr(ingest, "generate_embeddings", fail)
    result = ingest.run_ingestion()
    assert result["files_processed"] == []
    assert result["skipped"] == [indexed.name]


def test_model_dimension_change_uses_new_collection(indexed, monkeypatch):
    before = metadata_table.get_index_state()["active_collection"]
    monkeypatch.setattr(settings, "local_embedding_model", "another-model")
    with pytest.raises(ValueError, match="Index configuration"):
        chroma_manager.assert_compatible_index()
    monkeypatch.setattr(ingest, "generate_embeddings", lambda texts: [[0.1] * 32 for _ in texts])
    ingest.run_ingestion()
    assert metadata_table.get_index_state()["active_collection"] != before
    assert chroma_manager.similarity_search([0.1] * 32, 5)["documents"][0]


def test_missing_vector_collection_is_rebuilt(indexed, monkeypatch):
    state = metadata_table.get_index_state()
    chroma_manager._get_client().delete_collection(state["active_collection"])
    monkeypatch.setattr(chroma_manager, "_collection", None)
    ingest.run_ingestion()
    assert chroma_manager.count() > 0
    assert metadata_table.get_index_state()["active_collection"] != state["active_collection"]


def test_withdrawn_document_is_no_longer_searchable(indexed):
    indexed.unlink()  # disposable fixture PDF, never the repository corpus
    result = ingest.run_ingestion()
    assert result["removed"] == [indexed.name]
    assert chroma_manager.count() == 0
    assert metadata_table.get_ingested_files() == {}


def test_missing_source_mount_preserves_index(indexed, monkeypatch):
    before = metadata_table.get_index_state()
    monkeypatch.setattr(settings, "pdf_data_dir", str(indexed.parent / "missing-mount"))
    with pytest.raises(FileNotFoundError):
        ingest.run_ingestion()
    assert metadata_table.get_index_state() == before


def test_ingestion_does_not_require_unused_chat_key(sample_pdf, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(settings, "pdf_data_dir", str(sample_pdf.parent))
    assert ingest.run_ingestion()["total_chunks"] > 0


def test_rate_limiter_rejects_impossible_reservations():
    with pytest.raises(ValueError, match="quota"):
        SlidingWindowRateLimiter(10).acquire(32)
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)


def test_negative_bm25_does_not_hide_single_relevant_passage():
    from rag_pipeline.retrieval_service import RetrievedChunk, _hybrid_rescore
    chunk = RetrievedChunk("annual deductible coverage benefits", 0.9, "policy.pdf", "Policy", 1)
    assert _hybrid_rescore("annual deductible coverage benefits", [chunk])[0].score == 0.9


def test_filename_only_metadata_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "metadata_db_path", "metadata.sqlite")
    metadata_table.create_tables()
    assert metadata_table.count_chunks() == 0


def test_session_memories_are_independent():
    from frontend.session import get_session_pipeline
    first, second = {}, {}
    a, b = get_session_pipeline(first), get_session_pipeline(second)
    a.memory.add_turn("private question", "private answer")
    assert a is get_session_pipeline(first)
    assert a is not b
    assert b.memory.get_history() == []
    b.memory.clear()
    assert len(a.memory.get_history()) == 2


def test_mlflow_failure_does_not_stop_answer(monkeypatch):
    import mlflow.tracking
    from utils.mlflow_tracking import trace_query
    def fail(*args, **kwargs):
        raise OSError("unwritable tracking directory")
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", fail)
    with trace_query("question", 5, "model") as data:
        data["answer"] = "still available"
    assert data["answer"] == "still available"


def test_trace_does_not_swallow_application_exceptions(monkeypatch):
    from utils.mlflow_tracking import trace_query
    monkeypatch.setattr(settings, "enable_mlflow", False)
    with pytest.raises(RuntimeError, match="application"):
        with trace_query("question", 5, "model"):
            raise RuntimeError("application error")


def test_mlflow_records_successful_run():
    from pathlib import Path
    from mlflow.tracking import MlflowClient
    from utils.mlflow_tracking import trace_query
    with trace_query("question", 5, "model") as data:
        data["answer"] = "answer"
    client = MlflowClient(tracking_uri=Path(settings.mlflow_tracking_dir).resolve().as_uri())
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert "response_time_seconds" in runs[0].data.metrics


def test_streaming_failure_falls_back_to_normal_response(monkeypatch):
    from rag_pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    def fail(*args, **kwargs):
        raise RuntimeError("stream transport unsupported")
    monkeypatch.setattr(type(pipeline._llm), "stream", fail)
    monkeypatch.setattr(type(pipeline._llm), "invoke", lambda *a, **k: SimpleNamespace(content="A complete answer."))
    assert pipeline.generate_answer("context", "question", on_token=lambda text: None) == "A complete answer."


def test_empty_model_response_is_an_explicit_error(monkeypatch):
    from rag_pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    monkeypatch.setattr(type(pipeline._llm), "invoke", lambda *a, **k: SimpleNamespace(content=""))
    assert pipeline.generate_answer("context", "question")
    assert pipeline._generation_issue.code == "empty_response"


def test_content_blocks_are_supported(monkeypatch):
    from rag_pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    monkeypatch.setattr(type(pipeline._llm), "invoke", lambda *a, **k: SimpleNamespace(content=[{"type": "text", "text": "Answer."}]))
    assert pipeline.generate_answer("context", "question") == "Answer."


def test_generation_error_preserves_evidence_not_memory(indexed, monkeypatch):
    from rag_pipeline.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    def fail(*args, **kwargs):
        raise RuntimeError("401 invalid authentication credentials SECRET")
    monkeypatch.setattr(type(pipeline._llm), "invoke", fail)
    result = pipeline.answer_question("annual deductible")
    assert result["status"] == "error"
    assert result["error_code"] == "authentication"
    assert result["sources"][0]["excerpt"]
    assert result["confidence"] is None
    assert pipeline.memory.get_history() == []
    assert "SECRET" not in result["answer"]


def test_second_hop_failure_keeps_original_evidence(monkeypatch):
    import rag_pipeline.rag_pipeline as module
    from rag_pipeline.retrieval_service import RetrievedChunk
    pipeline = module.RAGPipeline()
    monkeypatch.setattr(settings, "enable_multi_hop", True)
    chunk = RetrievedChunk("The deductible is $250.", 0.9, "policy.pdf", "Policy", 1)
    def retrieve(query):
        if query == "next":
            raise RuntimeError("second search unavailable")
        return [chunk]
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(module, "plan_next_hop", lambda *a: "next")
    monkeypatch.setattr(type(pipeline._llm), "invoke", lambda *a, **k: SimpleNamespace(content="The deductible is $250."))
    result = pipeline.answer_question("deductible")
    assert result["status"] == "ok"
    assert result["warnings"]
    assert len(result["sources"]) == 1


def test_unsupported_amount_is_flagged():
    from rag_pipeline.guardrails import check_grounding
    assert not check_grounding("Your annual deductible is $9000.", ["Your annual deductible is $250."])
    assert check_grounding("Your annual deductible is $250.00. (Source: policy.pdf, page 99)", ["Your annual deductible is $250."])


def test_force_flag_reaches_ingestion(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(sys, "argv", ["ingest", "--force"])
    monkeypatch.setattr(ingest, "run_ingestion", lambda **kwargs: calls.append(kwargs) or {"files_processed": [], "removed": []})
    monkeypatch.setattr(chroma_manager, "count", lambda: 0)
    ingest.main()
    assert calls[0]["force"] is True


def test_relative_paths_do_not_depend_on_launch_directory(tmp_path, monkeypatch):
    from config.settings import _path_env, PROJECT_ROOT
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_TEST_PATH", raising=False)
    assert _path_env("RAG_TEST_PATH", "data/pdfs") == str(PROJECT_ROOT / "data/pdfs")


@pytest.mark.parametrize("name,value", [("top_k", 0), ("chunk_overlap", 1000), ("score_threshold", 1.1), ("request_timeout_seconds", 0)])
def test_invalid_settings_fail_early(name, value, monkeypatch):
    monkeypatch.setattr(settings, name, value)
    with pytest.raises(ValueError):
        settings.validate()


def test_reranking_evaluation_uses_equal_depth(monkeypatch):
    from scripts.evaluate_retrieval import _evaluate_one, _reranking_impact_summary
    from scripts.eval_questions import EvalQuestion
    from rag_pipeline.retrieval_service import RetrievedChunk
    chunks = [RetrievedChunk("text", 0.5, "policy.pdf", "Policy", page) for page in range(1, 7)]
    pipeline = SimpleNamespace(retrieve_with_stages=lambda q: SimpleNamespace(before_reranking=chunks, after_reranking=chunks[:5]))
    result = _evaluate_one(pipeline, EvalQuestion("question", "policy.pdf", 6))
    assert result.before_reranking.reciprocal_rank == result.after_reranking.reciprocal_rank == 0
    assert _reranking_impact_summary([result])["unchanged"] == 1
