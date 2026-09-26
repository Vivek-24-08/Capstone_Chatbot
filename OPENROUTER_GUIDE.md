# OpenRouter intelligent-search edition

This copy keeps the same installation, ingestion, and Streamlit commands as
CareGuide. OpenRouter generates chat answers. Embeddings run locally by default,
so indexing documents does not consume OpenRouter credits.

## Configure in a VS Code Bash terminal

From the cloned project directory, check the hidden configuration files. The
space in `ls -la .env*` is required.

```bash
ls -la .env*
```

If `.env` is not listed, create it from the template:

```bash
cp .env.example .env
ls -la .env*
code .env
```

If `code .env` is unavailable, refresh the VS Code Explorer and open `.env`
there. Add your key to `.env`, not `.env.example`:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_PROVIDER=local
ENABLE_INTELLIGENT_SEARCH=true
```

Use an exact model slug enabled for your OpenRouter account and approved by your
organization. Never commit `.env`.

Do not run `cat .env` or grep the key, because that displays the secret in the
terminal. This confirms that both required values loaded without printing them:

```bash
python3 -c "from config.settings import settings; print('OpenRouter configured:', bool(settings.openrouter_api_key and settings.openrouter_model))"
```

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
