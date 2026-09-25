# Running the capstone in a company sandbox

The installation and launch workflow is unchanged. Use Python 3.10–3.12 (the
Windows verification environment uses Python 3.12). From the repository root:

```bash
python -m venv venv
source venv/bin/activate       # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
# Edit .env with the provider settings and credentials approved for your sandbox.
python -m scripts.ingest
streamlit run frontend/app.py
```

In a managed notebook environment, activate/use the environment in which the
packages were installed when launching Streamlit. Keep your existing workspace
authentication or approved secret injection; do not commit credentials.

## If documents ingest but chat does not answer

Ingestion and answer generation are separate. Local embeddings can index files
even when the chat service is unavailable. A working embedding API also does not
prove that the chat model is enabled for your account.

The UI now displays the current stage and distinguishes authentication,
permission, model availability, quota, timeout, network, and index problems. If
generation fails after retrieval, the actual retrieved passages remain visible
under **View evidence**. They are evidence excerpts, not a generated answer.
Failed responses are excluded from conversation memory.

Gemini now defaults to REST over HTTPS (`GEMINI_TRANSPORT=rest`) for compatibility
with standard company proxies. Set `GEMINI_TRANSPORT=grpc` only when that transport
is supported by your network. TLS certificate verification remains enabled.

First run the read-only configuration/package/index checks:

```bash
python -m scripts.doctor
```

To test the configured embedding and chat services, make two small requests:

```bash
python -m scripts.doctor --probe
```

The probe can download a local embedding model on first use and may incur provider
usage. It never sends your policy documents. The sidebar's **Check AI connection**
button runs the same checks. Share the safe check results and the error category
with your sandbox administrator; never share `.env` or an API key.

| Error / symptom | Check |
|---|---|
| Credentials rejected | The active provider's credentials/identity; restart after changing `.env` |
| Permission denied | Model permissions and the sandbox's outbound access policy |
| Model unavailable | `GEMINI_CHAT_MODEL` or `DATABRICKS_LLM_ENDPOINT` must exist and be available to your account |
| Network / TLS / proxy error | Approved outbound access and your company's trusted certificates; do not disable TLS verification |
| Quota / delayed answer | Account quota; the local Gemini limiter also paces requests. Query rewriting consumes an additional call on follow-ups |
| Streaming fails | Automatic fallback uses a normal response; set `ENABLE_STREAMING=false` if the proxy consistently blocks streaming |
| Hugging Face downloads blocked | Use approved cached local models or an approved hosted embedding provider; `ENABLE_RERANKING=false` avoids the optional reranker download |
| Index configuration mismatch | Run `python -m scripts.ingest --force`, then restart or refresh the index |
| Tracking folder unavailable | Answers now continue; `ENABLE_MLFLOW=false` disables local tracing |
| No evidence found | Verify text extraction and the configured document folder; the app does not fabricate an answer |

Gemini requests use `REQUEST_TIMEOUT_SECONDS=45` by default. Application retries
are bounded; quota pacing and a streaming fallback can make total turn time longer
than one request timeout. Databricks request behavior also depends on its SDK and
endpoint configuration. The progress message remains visible while waiting.

Existing `.env` files remain compatible. New controls have defaults and do not
require additional setup steps. Relative paths now resolve against the repository
root, so CLI ingestion and Streamlit use the same database even if launched from
different working directories.

## Document updates and migration

Updates build a new Chroma collection before publishing it through a SQLite
transaction. Embedding or storage failures preserve the last published index.
Unchanged documents reuse vectors; provider/model or chunking changes rebuild the
index. Removing a PDF from an existing source folder withdraws it from the active
index. A missing source folder produces an error instead of withdrawing everything.

The first ingestion after this update rebuilds a legacy index to record its
configuration fingerprint. On an existing deployment, stop the old application
and back up its vector-store and metadata directories before installing the new
Chroma version. A new clone does not include those generated directories.

Prior collections remain physically on disk so in-flight readers are not broken.
Withdrawal removes content from active search, not from historical disk copies.
This is appropriate for the capstone; a production retention/cleanup policy is a
separate requirement. Do not run old and new application versions against the same
database concurrently.

The same commands still work:

```bash
python -m scripts.ingest
python -m scripts.ingest --force
pytest
python -m scripts.evaluate_retrieval
```

## What the panel can reasonably conclude

The interface labels scores as **retrieved-source relevance**, not answer accuracy.
The support check flags unsupported explicit monetary/percentage values and low
word overlap, but does not prove every claim or citation correct. Validate the
answers used in the demonstration against the PDFs. Offline regression tests use
fake embeddings and mocked model responses; a live connection check in the actual
company sandbox is still necessary.
