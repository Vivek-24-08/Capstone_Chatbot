# Sandbox reliability validation

Validated on Windows with Python 3.12 in an isolated virtual environment.

- Installation from `requirements.txt` completed successfully.
- `pip check`: no broken requirements found.
- Regression suite: **123 passed**, with two protobuf deprecation warnings.
- Streamlit AppTest covers missing credentials, successful streamed answers, and
  authentication failure after document retrieval.
- Regression tests cover failed ingestion rollback, index configuration changes,
  document withdrawal, per-session conversation isolation, streaming fallback,
  empty responses, and failures in optional tracing.
- Gemini transport tests verify timeout and retry arguments reach the SDK call.
- SentenceTransformer and CrossEncoder imports succeeded.

Model calls in these tests use controlled substitutes. Live Gemini/Databricks
credentials, company proxy access, model downloads in the company sandbox, and
answer quality on your presentation documents have not been verified.

The installation and launch commands remain unchanged. Chroma was updated to
1.0.20 because the previous pinned version required a C++ build toolchain on this
Windows/Python setup. Follow the backup guidance in SANDBOX_GUIDE.md when upgrading
an existing database.

Run `python -m scripts.doctor --probe` in the company sandbox to check the actual
embedding and chat connections. This makes small service requests, but does not
send policy documents.

Base revision: `9d8f84450bc4d3a4728c868da8650cd1799fce7f`.
Improvement branch: `codex/sandbox-reliability`.
