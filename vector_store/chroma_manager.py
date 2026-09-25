"""Persistent vector search with an atomically published collection pointer."""
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from vector_store import metadata_table
from vector_store.index_config import index_fingerprint

_client = None
_collection = None
_client_path = None
_collection_key = None

def _get_client():
    global _client, _client_path
    if _client is None or _client_path != settings.chroma_persist_dir:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir,
                                           settings=ChromaSettings(anonymized_telemetry=False))
        _client_path = settings.chroma_persist_dir
    return _client

def _get_collection():
    global _collection, _collection_key
    state = metadata_table.get_index_state()
    name = state["active_collection"] if state else settings.chroma_collection_name
    key = (settings.chroma_persist_dir, name)
    if _collection is None or key != _collection_key:
        client = _get_client()
        if state:
            # A missing published index is corruption, not an empty knowledge base.
            _collection = client.get_collection(name, embedding_function=None)
        else:
            _collection = client.get_or_create_collection(name, embedding_function=None,
                                                          metadata={"hnsw:space": "cosine"})
        _collection_key = key
    return _collection

def assert_compatible_index():
    state = metadata_table.get_index_state()
    if state and state["fingerprint"] != index_fingerprint():
        raise ValueError("Index configuration differs from the current embedding space.")
    return "Index configuration matches." if state else "Legacy or empty index; ingestion will initialize its fingerprint."

def _upsert(collection, chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts differ.")
    # Chroma limits write batch sizes; do not send a whole long PDF in one call.
    for start in range(0, len(chunks), 256):
        batch = chunks[start:start + 256]
        collection.upsert(ids=[c.chunk_id for c in batch],
                          embeddings=embeddings[start:start + 256],
                          documents=[c.chunk_text for c in batch],
                          metadatas=[{"document_name": c.file_name, "document_type": c.document_type,
                                      "page_number": c.page_number, "created_timestamp": c.created_timestamp,
                                      "chapter_title": c.chapter_title or "", "section_title": c.section_title or ""}
                                     for c in batch])

def upsert_chunks(chunks, embeddings):
    _upsert(_get_collection(), chunks, embeddings)

def stage_generation(chunks, embeddings):
    name = settings.chroma_collection_name[:40] + "-" + uuid.uuid4().hex[:16]
    client = _get_client()
    collection = client.create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})
    try:
        _upsert(collection, chunks, embeddings)
        if collection.count() != len(chunks):
            raise RuntimeError("Staged index count does not match the prepared chunks.")
    except BaseException:
        client.delete_collection(name)
        raise
    return name

def discard_generation(name):
    _get_client().delete_collection(name)

def read_document(document_name):
    return _get_collection().get(where={"document_name": document_name}, include=["documents", "metadatas", "embeddings"])

def delete_by_document(document_name):
    _get_collection().delete(where={"document_name": document_name})

def similarity_search(query_embedding, top_k, where=None):
    assert_compatible_index()
    collection = _get_collection()
    size = collection.count()
    if size == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    return collection.query(query_embeddings=[query_embedding], n_results=min(top_k, size), where=where)

def count():
    return _get_collection().count()
