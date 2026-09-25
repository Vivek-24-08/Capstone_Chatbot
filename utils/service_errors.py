"""Safe, actionable messages. Never display raw provider exceptions or keys."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceIssue:
    code: str
    message: str
    retryable: bool = False


def classify_error(exc: BaseException) -> ServiceIssue:
    text = str(exc).lower()
    if "pdf directory" in text:
        return ServiceIssue("documents", "The document folder is missing. Check PDF_DATA_DIR and the sandbox's mounted folders.")
    if "no readable text" in text:
        return ServiceIssue("documents", "A document has no readable text. Use a text-based PDF or apply OCR before indexing. The previous index has been preserved.")
    if "empty response" in text:
        return ServiceIssue("empty_response", "The model returned no answer text. Try rephrasing the question; if this repeats, check the configured model with python -m scripts.doctor --probe.")
    if any(s in text for s in ("unauthenticated", "invalid authentication", "api key not valid", "api_key_invalid", "invalid api key", "default credentials", "401")):
        return ServiceIssue("authentication", "The AI service rejected its credentials. Check the active provider's credentials in .env and restart the app.")
    if any(s in text for s in ("permission_denied", "permission denied", "403", "forbidden")):
        return ServiceIssue("permission", "The AI service denied access. Check model permissions and your company's network policy.")
    if any(s in text for s in ("404", "not_found", "not found for api", "not supported for generate", "model not found")):
        return ServiceIssue("model", "The configured AI model or endpoint is unavailable. Check GEMINI_CHAT_MODEL or DATABRICKS_LLM_ENDPOINT for this account, then restart.")
    if any(s in text for s in ("429", "resource_exhausted", "quota", "rate limit")):
        return ServiceIssue("quota", "The AI service is rate-limited or its quota is exhausted. Wait before retrying; if this continues, check the account quota.", True)
    if any(s in text for s in ("timeout", "timed out", "deadline")):
        return ServiceIssue("timeout", "The AI service did not respond within the time limit. Retry, or check whether your sandbox can reach the configured endpoint.", True)
    if any(s in text for s in ("ssl", "certificate", "proxy", "connection", "dns", "resolve", "unreachable")):
        return ServiceIssue("network", "The sandbox cannot reach the AI service. Check outbound access, proxy settings, and your company's trusted certificates.", True)
    if any(s in text for s in ("dimension", "embedding space", "index configuration")):
        return ServiceIssue("index", "The index does not match the embedding configuration. Run python -m scripts.ingest --force with the current .env, then restart the app.")
    if any(s in text for s in ("readonly", "read-only", "unable to open database", "disk")):
        return ServiceIssue("storage", "The application cannot access its local storage. Check write access to the configured vector store and metadata paths.")
    return ServiceIssue("service", "The request could not be completed. Retry once; if it continues, run python -m scripts.doctor --probe and check the server log.")


def is_transient(exc: BaseException) -> bool:
    return classify_error(exc).retryable or any(s in str(exc).lower() for s in ("503", "unavailable", "502"))
