# Healthcare Insurance Assistant — Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers member, provider,
and policy questions using your official insurance plan documents (Evidence
of Coverage, Summary of Benefits, and related policy manuals). Built to run
with a local application and local document storage. Answer generation uses the
configured OpenRouter service by default; local embeddings do not make chat offline.

This edition adds an explainable intelligent-search router for insurance topics
and uses OpenRouter's OpenAI-compatible HTTPS endpoint for answer generation.
See [OPENROUTER_GUIDE.md](OPENROUTER_GUIDE.md) for the short sandbox setup.

**Sandbox reliability update:** installation and launch commands are unchanged.
See [SANDBOX_GUIDE.md](SANDBOX_GUIDE.md) for connection checks, error categories,
safe index updates, and the steps to use when documents ingest but chat fails.
Optional diagnostics: `python -m scripts.doctor --probe` (makes small API requests).

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (Python)                            │
│                                                                           │
│  data/pdfs/*.pdf                                                         │
│        │                                                                 │
│        ▼                                                                 │
│  ingestion/pdf_loader.py        [PyMuPDF: load + parse, page-by-page]   │
│        │                                                                 │
│        ▼                                                                 │
│  chunking/chunker.py            [chapter/section-aware for structured   │
│        │                          docs (structure_chunker.py), else     │
│        │                          RecursiveCharacterTextSplitter 1000/200│
│        │                          + file/page/type/chapter/section meta]│
│        ▼                                                                 │
│  embeddings/embedding_service.py [pluggable: local (sentence-transformers)│
│        │                          or Gemini (gemini-embedding-001)]      │
│        ▼                                                                 │
│  vector_store/                                                          │
│    chroma_manager.py            [ChromaDB: similarity search index]     │
│    metadata_table.py            [SQLite: "Delta Table" analog, audit +  │
│                                    idempotent re-ingestion tracking]     │
│        │                                                                 │
│        ▼                                                                 │
│  rag_pipeline/                                                          │
│    retrieval_service.py    top-K + score threshold + hybrid (BM25)      │
│    search_intelligence.py  insurance intent + query expansion + weights │
│    multi_query.py           optional: paraphrase + fuse (RRF)           │
│    reranker.py              optional cross-encoder re-scoring           │
│    query_rewriter.py        follow-up question -> standalone question   │
│    multi_hop.py             optional: follow-up retrieval round         │
│    memory.py                 capped conversation history                │
│    guardrails.py            input/output safety checks                  │
│    prompt_templates.py      grounded system prompt + citation format    │
│    rag_pipeline.py           RAGPipeline: retrieve→context→generate     │
│        │                                                                 │
│        ▼                                                                 │
│  OpenRouter Chat Model                 ──► grounded, cited answer       │
│        │                                                                 │
└────────┼──────────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND: frontend/app.py (Streamlit)                │
│         Chat box only — no upload UI. Ingestion is fully backend-owned.  │
│         Shows: answer, expandable source citations, confidence, a        │
│         grounding-check warning when flagged, sidebar (model info,       │
│         indexed chunk count), clear-chat.                                 │
└─────────────────────────────────────────────────────────────────────────┘

Observability: utils/mlflow_tracking.py logs every question as a local
MLflow run (params, latency, token estimates, retrieved sources) — browse
with `mlflow ui` from the project root.
```

## 2. Why these choices (and what changed from a generic template)

This project combines a Databricks/Azure-OpenAI-flavored architecture with a
local-first application. OpenRouter supplies the chat model through one
OpenAI-compatible endpoint, while local equivalents replace managed storage:

| Spec asked for | This project uses | Why |
|---|---|---|
| Databricks Vector Search | **ChromaDB** (on-disk) | Zero setup, no cluster, same "index + similarity search" role |
| Databricks Delta Table | **SQLite** (`vector_store/metadata_table.py`) | Same schema, same structured/queryable role, zero server |
| Databricks Secret Scope | **`.env` file** (gitignored) | `config/settings.py` isolates all secret reads to one file, so swapping the *source* later is a one-file change |
| Azure OpenAI / `text-embedding-3-large` | **Pluggable embeddings**: local `sentence-transformers`, Gemini, or Databricks | Local embeddings are the default and do not consume OpenRouter credits |
| MLflow on a Databricks tracking server | **MLflow, local file store** (`mlruns/`) | Identical `mlflow.log_*` API; only the tracking URI differs |

## 3. Setup (unrestricted machine — recommended path)

```bash
git clone <this repo>
cd rag_chatbot
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
# ^ installs a CPU-only build of torch (~3 GB total install) via the
#   --extra-index-url pin at the top of requirements.txt -- without it, pip
#   would pull a GPU/CUDA build (~7 GB) that this project never uses, since
#   local embeddings and reranking only ever run on CPU here.

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY and an exact OPENROUTER_MODEL slug.
# Leave EMBEDDING_PROVIDER=local so indexing does not consume API credits.

python -m scripts.ingest           # one-time: indexes data/pdfs/*.pdf
streamlit run frontend/app.py      # opens the chat UI in your browser
```

### VS Code sandbox setup

Run these commands from the VS Code Bash terminal. There must be a space after
`-la`; the correct command is `ls -la .env*`, not `ls -la.env`.

```bash
cd ~/Capstone_Chatbot_OpenRouter

# See whether the template and local configuration already exist.
ls -la .env*

# Create .env only when the preceding output does not already show .env.
cp .env.example .env

# Confirm both files exist. This lists names and metadata, not their contents.
ls -la .env*

# Open the local configuration in VS Code.
code .env
```

If `code .env` is unavailable, refresh the VS Code Explorer and click `.env`.
Set these values and save with Ctrl+S:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_actual_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_PROVIDER=local
ENABLE_INTELLIGENT_SEARCH=true
```

Do not use `cat .env` or `grep OPENROUTER_API_KEY .env`, because those commands
print the secret. Verify the configuration without displaying it:

```bash
python3 -c "from config.settings import settings; print('OpenRouter configured:', bool(settings.openrouter_api_key and settings.openrouter_model))"
```

The result should be `OpenRouter configured: True`. Then continue:

```bash
python3 -m scripts.ingest
python3 -m scripts.doctor --probe
streamlit run frontend/app.py
```

Add your own PDFs by dropping them into `data/pdfs/` and re-running
`python -m scripts.ingest` (or just restarting the Streamlit app — it
auto-detects new/changed files via a content hash and only processes what's
new).

## 4. Switching embedding providers (Phase 1 → Phase 2)

Everything routes through `embeddings/embedding_service.py`, so this is a
one-line change in `.env`:

```bash
EMBEDDING_PROVIDER=local     # free, offline, sentence-transformers (default)
EMBEDDING_PROVIDER=gemini    # hosted, uses your GEMINI_API_KEY, no model download
```

**Important:** switching providers changes the vector space. Ingestion now detects
provider/model and chunking changes and builds a fresh collection before switching
the active index. `python -m scripts.ingest --force` also requests a full rebuild.
Do not manually delete the database to switch providers.

## 5. Choosing which local embedding model to use

`LOCAL_EMBEDDING_MODEL` defaults to `sentence-transformers/all-mpnet-base-v2`.
You can compare it with smaller CPU models using the evaluation below. Rather than trust
a public leaderboard (which measures general web/Wikipedia text, not
insurance-specific language), this project includes its own small
MTEB-style evaluation tool that measures retrieval quality against your
**actual** ingested documents:

```bash
python -m scripts.evaluate_embeddings
```

This downloads a handful of candidate models (see `CANDIDATE_MODELS` at the
top of `scripts/evaluate_embeddings.py` — edit that list to try others),
embeds your real chunks and a set of hand-labeled real questions
(`EVAL_QUESTIONS`, shared from `scripts/eval_questions.py`), and reports
Recall@1/3/5, Precision@1/3/5, NDCG@1/3/5, and MRR per model, ending with a
recommendation and the exact `.env` line to apply it. Needs
`huggingface.co` reachable (won't run in the network-restricted sandbox
this project was partly built in — see section 7). After switching models,
re-ingest as described above. Output is printed only, not saved anywhere.

**Important:** this evaluates raw embedding-model similarity only — no
hybrid search, reranking, or score threshold, so it doesn't reflect what a
real user actually gets back. To evaluate the full, deployed retrieval
pipeline (the one real questions actually go through) against the same
ground truth, run:

```bash
python -m scripts.evaluate_retrieval   # needs the vector store already populated
```

This uses whichever provider/settings are currently in `.env`, reports the
same four metrics **twice — once BEFORE reranking (vector/hybrid search
only) and once AFTER** — plus a one-line summary of how many questions
reranking actually improved, left unchanged, or made worse. That's the only
way to see reranking's real effect: `RAGPipeline.retrieve()` on its own only
ever returns the final, post-reranking list, so without this the
pre-reranking ranking is invisible. If `ENABLE_RERANKING=false`, the two
stages are identical (nothing to compare) and the impact summary is
skipped.

Unlike the embedding-model comparison above, this also saves every
individual question's before/after result (not just the average) to
`eval_results/retrieval_eval_<timestamp>.json`, so a specific regression, or
reranking actively hurting one particular question, can be traced back by
name. It also prints which questions missed the correct page within
`TOP_K` after reranking, for quick debugging. The underlying metric
formulas live in `utils/retrieval_metrics.py`, shared by
both scripts so their numbers are directly comparable.

**Reviewing metrics in git, without CI:** every run also overwrites
`eval_results/latest.json` — the one snapshot in that folder that ISN'T
gitignored, specifically so it can be committed and diffed in a PR like any
other file. It also records a `corpus_fingerprint` (a hash of every file in
`data/pdfs/`) plus the active `embedding_model` and `reranker_model`, so
anyone reviewing a commit of this file can tell at a glance whether a change
in the numbers was caused by different documents or a different model. This
is deliberately a manual step, not automatic: re-run
`python -m scripts.evaluate_retrieval` and commit `eval_results/latest.json`
whenever you change `data/pdfs/`, `RERANKER_MODEL`, or the embedding
provider/model, the same way you'd update any other generated file that
depends on project state. Wiring an actual CI trigger for this (so it
happens on every relevant PR rather than being something you remember to
do) is a reasonable next step, but is a separate, bigger change (needs
`GEMINI_API_KEY` as a repo secret and network access in CI) that hasn't been
built here.

## 6. Fully keyless mode: replacing Gemini with Databricks-hosted models

Every LLM API normally requires your app to hold a secret credential. This
project supports one deployment mode that genuinely doesn't: running inside
Databricks itself, calling Databricks Foundation Model APIs.

```bash
LLM_PROVIDER=databricks
EMBEDDING_PROVIDER=databricks
# DATABRICKS_HOST / DATABRICKS_TOKEN can stay blank when this app runs
# inside a Databricks notebook, job, or App -- auth is then fully automatic.
```

With both switches set, `GEMINI_API_KEY` is never read (`config/settings.py`
only requires it when a provider is actually `"gemini"`) — Gemini is
completely out of the runtime path. `rag_pipeline/llm_service.py` and
`embeddings/embedding_service.py` are the two factories that dispatch on
these switches; nothing else in the app needs to change.

**Tested vs. not tested:** the provider-dispatch logic itself is covered by
`tests/test_provider_switching.py` (mocked, no real endpoint). Actually
calling a live Databricks Model Serving endpoint could **not** be tested in
this project's own build sandbox — there's no Databricks workspace attached
there. Smoke-test `LLM_PROVIDER=databricks` against your real workspace
before relying on it.

**Dependency note:** `requirements.txt` pins the older `langchain-databricks`
package rather than the newer `databricks-langchain`, because the newer one
pulls in `langchain-core` 1.x, which conflicts with `langchain-google-genai`
and with the MLflow version this project's local file-store tracking depends
on. If you go fully keyless (drop Gemini entirely), you can safely upgrade
to `databricks-langchain` and remove the Gemini-specific pins.

## 7. A note on this project's own development/test sandbox

This app was built and smoke-tested inside a network-restricted cloud
sandbox. Two sandbox-specific constraints are worth knowing about if you hit
them in a similar restricted environment (they do **not** apply on a normal
laptop/server with open internet access):

- **`huggingface.co` may be blocked** by an egress policy in some sandboxes.
  The `local` embedding provider downloads its model from the Hugging Face
  Hub on first use, so it will fail there — use `EMBEDDING_PROVIDER=gemini`
  instead in that environment. The `reranker` (cross-encoder) has the same
  dependency and degrades gracefully (logs a warning, skips reranking)
  rather than crashing if it can't download its model.
- **No browser/port-forwarding was available** in that sandbox, so the
  Streamlit UI itself could only be smoke-tested for "does the process boot
  and serve HTTP" (`curl localhost:8501`), not a real interactive
  walkthrough. A live, interactive test of the chat UI should be done on
  your own machine or any environment with normal port access.
- OpenRouter model availability depends on your account and provider routing.
  If `OPENROUTER_MODEL` is rejected, select an exact model slug from the
  OpenRouter model catalog and update `.env`.
- The Gemini free tier caps embedding calls at roughly 100/minute, and each
  text in a batch counts individually against that quota. `scripts/ingest.py`
  paces itself under this limit automatically when `EMBEDDING_PROVIDER=gemini`
  (see `embeddings/gemini_embeddings.py`'s rate limiter) — ingesting a few
  hundred chunks may take several minutes on the free tier. This does not
  apply to the `local` provider, which has no API rate limit.

## 8. Testing

```bash
pytest
```

The full suite (`tests/`) runs offline and deterministically: `conftest.py`
swaps in a fake, hash-based embedding provider and points every on-disk
store (Chroma, SQLite, MLflow) at a fresh temp directory per test, and the
tests that reach chat models use controlled substitutes — no real API key or
network call is exercised by the test suite.

## 9. Enhancements already built in

- **Conversational memory** (`rag_pipeline/memory.py`) — capped chat history for natural follow-ups.
- **Hybrid search** (`rag_pipeline/retrieval_service.py`) — vector + BM25 keyword blend, toggle via `ENABLE_HYBRID_SEARCH`.
- **Intelligent search routing** (`rag_pipeline/search_intelligence.py`) — detects insurance intent, expands the keyword branch with domain terms, and adjusts semantic/keyword weights without adding an API call.
- **Query rewriting** (`rag_pipeline/query_rewriter.py`) — follow-ups rewritten into standalone questions before retrieval.
- **Reranking** (`rag_pipeline/reranker.py`) — optional CPU cross-encoder re-scoring, toggle via `ENABLE_RERANKING`.
- **Feedback logging** (`utils/feedback_logger.py`) — 👍/👎 buttons in the UI append to a local `.jsonl` file.
- **Guardrails** (`rag_pipeline/guardrails.py`) — prompt-injection denylist + lexical grounding check on every answer.
- **Source citations + confidence** — every answer shows an expandable source panel and a retrieval-based confidence score.
- **Follow-up handling** — covered jointly by memory + query rewriting above.
- **Multi-query retrieval** (`rag_pipeline/multi_query.py`) — searches with several LLM-generated paraphrasings of the question and fuses the results via reciprocal rank fusion, so retrieval isn't only as good as the user's exact wording. Off by default (`ENABLE_MULTI_QUERY`) because it costs an extra chat call per question.
- **Multi-hop retrieval** (`rag_pipeline/multi_hop.py`) — after the first retrieval pass, lets the model ask itself a follow-up search query when a compound question needs a second, different piece of information, then merges both rounds' chunks before answering. Off by default (`ENABLE_MULTI_HOP`), bounded by `MAX_HOPS`.
- **Retrieval evaluation** (`utils/retrieval_metrics.py`, `scripts/evaluate_embeddings.py`, `scripts/evaluate_retrieval.py`) — Recall@K, Precision@K, NDCG@K, and MRR against hand-labeled ground truth, for both a candidate embedding model in isolation and the full deployed pipeline — see section 5.
- **Streaming answers** (`rag_pipeline/rag_pipeline.py`'s `on_token` callback, used in `frontend/app.py`) — the answer renders token-by-token as OpenRouter returns it, instead of appearing all at once after the full response completes. This improves perceived latency only; retrieval, grounding, memory, and MLflow logging are unaffected.
- **Accessible display controls** — application zoom from 85–135%, high contrast, reduced motion, and optionally expanded evidence are available in the sidebar. Browser zoom shortcuts continue to work.
- **Concurrent multi-query retrieval** (`rag_pipeline.py`'s `retrieve_with_stages()`) — when `ENABLE_MULTI_QUERY` is on, its independent per-variant searches run in a thread pool instead of one after another, cutting that feature's added latency roughly to the slowest single search instead of their sum.

## 10. Path to a real Databricks/production deployment

Nothing in this codebase needs to change structurally to move to Databricks
— only the modules explicitly called out in section 2's table get swapped.
The LLM and embedding swap (item 0 below) is already done — see section 6.

0. Chat + embeddings → set `LLM_PROVIDER=databricks` and
   `EMBEDDING_PROVIDER=databricks` (section 6). Already implemented and
   config-driven; no code change needed for this part.
1. `vector_store/chroma_manager.py` → a thin wrapper around a Databricks
   Vector Search endpoint + index (`databricks-vectorsearch` SDK), synced
   from a real Delta table.
2. `vector_store/metadata_table.py` → the SQLite calls become Delta table
   writes/reads via Spark or the Databricks SQL connector, using the exact
   same schema already defined here.
3. `config/settings.py` → read secrets via `dbutils.secrets.get(scope, key)`
   instead of `os.getenv`, since it's the only file that touches secrets.
4. `utils/mlflow_tracking.py` → change one line,
   `mlflow.set_tracking_uri(...)`, to point at the workspace-hosted tracking
   server instead of a local `mlruns/` folder; every `mlflow.log_*` call is
   unchanged.
5. Deploy `frontend/app.py` as a Databricks App (or any standard Streamlit
   host) instead of running it locally.

## 11. Project structure

```
rag_chatbot/
├── data/pdfs/                  # source PDFs (backend-managed, no upload UI)
├── ingestion/pdf_loader.py
├── chunking/                    # chunker.py (dispatch), structure_chunker.py (chapter/section-aware)
├── embeddings/                 # base.py, local_embeddings.py, gemini_embeddings.py, embedding_service.py
├── vector_store/                # chroma_manager.py, metadata_table.py
├── rag_pipeline/                # retrieval_service, multi_query, reranker, query_rewriter, multi_hop, memory, guardrails, prompt_templates, rag_pipeline
├── frontend/app.py              # Streamlit chat-only UI
├── config/settings.py
├── utils/                       # logging_utils, mlflow_tracking, feedback_logger
├── scripts/ingest.py            # CLI + auto-ingest entry point
├── scripts/evaluate_embeddings.py, evaluate_retrieval.py, eval_questions.py  # evaluation tooling
├── utils/retrieval_metrics.py   # Recall@K, Precision@K, NDCG@K, MRR
├── tests/                       # pytest suite
├── conftest.py                  # shared fixtures (fake embeddings, temp stores)
├── requirements.txt
├── .env.example
└── README.md
```
