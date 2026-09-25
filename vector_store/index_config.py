"""Fingerprint the embedding space and the text preparation rules."""
import hashlib
import json
from config.settings import settings


def index_fingerprint():
    models = {"local": settings.local_embedding_model, "gemini": settings.gemini_embedding_model,
              "databricks": settings.databricks_embedding_endpoint}
    config = {"schema": 1, "provider": settings.embedding_provider,
              "model": models[settings.embedding_provider],
              "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
