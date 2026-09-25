"""Best-effort tracing; logging failures must not prevent answers."""
from contextlib import contextmanager
from pathlib import Path
import time
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def _estimate_tokens(text):
    return max(1, len(text) // 4)

@contextmanager
def trace_query(question, top_k, chat_model):
    start = time.perf_counter()
    data = {"answer": "", "retrieved_chunks": []}
    client = run_id = None
    if settings.enable_mlflow:
        try:
            from mlflow.tracking import MlflowClient
            folder = Path(settings.mlflow_tracking_dir).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            client = MlflowClient(tracking_uri=folder.as_uri())
            experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
            if experiment is None:
                try:
                    experiment_id = client.create_experiment(settings.mlflow_experiment_name)
                except Exception:
                    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
                    if experiment is None:
                        raise
                    experiment_id = experiment.experiment_id
            else:
                experiment_id = experiment.experiment_id
            # Explicit IDs avoid global active-run state across browser threads.
            run_id = client.create_run(experiment_id).info.run_id
            for key, value in {"chat_model": chat_model, "embedding_provider": settings.embedding_provider, "top_k": top_k}.items():
                client.log_param(run_id, key, value)
            client.log_text(run_id, question, "question.txt")
        except Exception:
            logger.warning("Query tracing unavailable; answering continues.", exc_info=True)
    failed = False
    try:
        yield data
    except BaseException:
        failed = True
        raise
    finally:
        if client is not None and run_id is not None:
            try:
                client.log_metric(run_id, "response_time_seconds", time.perf_counter() - start)
                client.log_metric(run_id, "num_chunks_retrieved", len(data.get("retrieved_chunks", [])))
                client.log_metric(run_id, "question_tokens_estimate", _estimate_tokens(question))
                client.log_metric(run_id, "completion_tokens_estimate", _estimate_tokens(data.get("answer", "")))
                client.log_text(run_id, data.get("answer", ""), "answer.txt")
                client.log_dict(run_id, {"sources": data.get("retrieved_chunks", [])}, "retrieved_sources.json")
            except Exception:
                logger.warning("Could not save query trace; answer is unaffected.", exc_info=True)
            finally:
                try:
                    client.set_terminated(run_id, "FAILED" if failed else "FINISHED")
                except Exception:
                    logger.warning("Could not close query trace.", exc_info=True)
