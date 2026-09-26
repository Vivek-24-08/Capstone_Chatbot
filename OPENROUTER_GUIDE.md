# OpenRouter intelligent-search edition

This copy keeps the same installation, ingestion, and Streamlit commands as
CareGuide. OpenRouter generates chat answers. Embeddings run locally by default,
so indexing documents does not consume OpenRouter credits.

## Configure

Copy `.env.example` to `.env`, then set:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_PROVIDER=local
ENABLE_INTELLIGENT_SEARCH=true
```

Use an exact model slug enabled for your OpenRouter account and approved by your
organization. Never commit `.env`.

## Run

```bash
pip3 install --user -r requirements.txt
python3 -m scripts.ingest
python3 -m scripts.doctor --probe
streamlit run frontend/app.py
```

`doctor --probe` makes one small embedding request and one small chat request.
It does not send the indexed policy documents.

## Intelligent search

The search layer remains explainable and deterministic. It:

1. rewrites conversational follow-ups into standalone questions;
2. detects insurance topics such as costs, pharmacy, claims, appeals, provider
   networks, exclusions, and coverage;
3. expands only the keyword-search branch with related insurance terminology;
4. adjusts semantic-versus-keyword weights based on the detected topic;
5. reranks the candidate passages before answer generation.

The original user question still drives semantic vector search. This avoids
letting query expansion distort its meaning. All final answers remain constrained
to retrieved document passages and show their evidence.
