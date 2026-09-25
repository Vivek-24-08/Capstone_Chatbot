"""Incremental ingestion with staged vectors and an atomic publish step.

Existing commands are unchanged: python -m scripts.ingest [--force].
Unchanged documents reuse vectors; incompatible embedding configurations rebuild.
Old collections remain on disk so in-flight readers can finish safely.
"""
import argparse
import hashlib
import math
from pathlib import Path
from filelock import FileLock
from chunking.chunker import Chunk, chunk_documents
from config.settings import settings
from embeddings.embedding_service import generate_embeddings
from ingestion.pdf_loader import _extract_pages_from_pdf
from vector_store import chroma_manager, metadata_table
from vector_store.index_config import index_fingerprint
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()

def _validate_vectors(chunks, vectors):
    if len(chunks) != len(vectors) or not vectors:
        raise ValueError("Embedding provider returned an incomplete batch.")
    dimension = len(vectors[0])
    if not dimension or any(len(v) != dimension or not all(math.isfinite(float(x)) for x in v) for v in vectors):
        raise ValueError("Embedding provider returned invalid vectors.")

def _reuse_document(name):
    raw = chroma_manager.read_document(name)
    chunks = [Chunk(chunk_id=identifier, chunk_text=text, file_name=name,
                    page_number=meta["page_number"], document_type=meta["document_type"],
                    created_timestamp=meta["created_timestamp"], chapter_title=meta.get("chapter_title"),
                    section_title=meta.get("section_title"))
              for identifier, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    embeddings = [list(map(float, vector)) for vector in raw["embeddings"]]
    return chunks, embeddings

def run_ingestion(force=False, on_status=None):
    settings.validate(require_chat=False)
    folder = Path(settings.pdf_data_dir)
    if not folder.is_dir():
        # An unavailable/mistyped mount must never withdraw every document.
        raise FileNotFoundError("PDF directory is missing; check PDF_DATA_DIR.")
    Path(settings.metadata_db_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.metadata_db_path) + ".ingest.lock", timeout=10):
        return _run_locked(folder, force, on_status or (lambda message: None))

def _run_locked(folder, force, notify):
    metadata_table.create_tables()
    paths = sorted(folder.glob("*.pdf"))
    hashes = {p.name: _file_hash(p) for p in paths}
    previous = metadata_table.get_ingested_files()
    state = metadata_table.get_index_state()
    fingerprint = index_fingerprint()
    reuse = not force and bool(state) and state["fingerprint"] == fingerprint
    try:
        healthy = chroma_manager.count() == sum(info["chunk_count"] for info in previous.values())
    except Exception:
        healthy = False
    reuse = reuse and healthy
    removed = sorted(set(previous) - set(hashes))
    changed = [p.name for p in paths if not reuse or previous.get(p.name, {}).get("file_hash") != hashes[p.name]]
    if reuse and not changed and not removed:
        return {"files_processed": [], "total_chunks": 0, "skipped": list(hashes), "removed": []}
    all_chunks, all_vectors, file_info = [], [], {}
    for path in paths:
        if path.name not in changed:
            chunks, vectors = _reuse_document(path.name)
            if len(chunks) != previous[path.name]["chunk_count"]:
                raise RuntimeError("Index document counts are inconsistent; retry ingestion with --force.")
        else:
            notify(f"Indexing {path.name}...")
            pages = _extract_pages_from_pdf(path)
            chunks = chunk_documents(pages)
            if not chunks:
                raise ValueError(f"No readable text in {path.name}; use a text-based PDF or OCR it first. Previous index preserved.")
            vectors = generate_embeddings([c.chunk_text for c in chunks])
            _validate_vectors(chunks, vectors)
        all_chunks.extend(chunks)
        all_vectors.extend(vectors)
        file_info[path.name] = {"hash": hashes[path.name], "count": len(chunks)}
    # Catch edits/removals/additions while processing; never publish stale file hashes.
    if {p.name: _file_hash(p) for p in sorted(folder.glob("*.pdf"))} != hashes:
        raise RuntimeError("Source PDFs changed during ingestion. Retry; the previous index is preserved.")
    notify("Publishing the prepared index...")
    name = chroma_manager.stage_generation(all_chunks, all_vectors)
    try:
        metadata_table.publish_generation(name, fingerprint, all_chunks, all_vectors, file_info)
    except BaseException:
        chroma_manager.discard_generation(name)
        raise
    return {"files_processed": changed,
            "total_chunks": sum(file_info[n]["count"] for n in changed),
            "skipped": [n for n in hashes if n not in changed], "removed": removed}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild all vectors in a fresh collection before publishing.")
    args = parser.parse_args()
    result = run_ingestion(force=args.force, on_status=print)
    print(f"Files processed: {result['files_processed']}")
    print(f"Files withdrawn: {result['removed']}")
    print(f"Total chunks now in index: {chroma_manager.count()}")

if __name__ == "__main__":
    main()
