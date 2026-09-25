# ==============================================================================
# rag_pipeline/rag_pipeline.py
# ------------------------------------------------------------------------------
# THE CENTRAL ORCHESTRATOR: wires retrieval, reranking, guardrails, prompting,
# and the Gemini chat model into one class the Streamlit frontend calls.
#
# WHAT THIS FILE DOES
#   Defines RAGPipeline, with four methods matching the spec's required
#   method names -- each one a single, clear stage of "Retrieve Top K
#   Chunks -> Ground LLM Response Using Retrieved Context -> Return Answer
#   with Citations":
#
#     retrieve(question)        Step: vector (+hybrid+multi-query+rerank) search
#     build_context(chunks)     Step: format chunks into a citation-ready
#                                block of text for the prompt
#     generate_answer(...)      Step: call Gemini with the grounded prompt
#     answer_question(question) The single public entry point that runs the
#                                whole pipeline end-to-end for one turn,
#                                including the optional multi-hop follow-up
#                                round between build_context and generate_answer
#
# WHY SPLIT INTO FOUR METHODS INSTEAD OF ONE BIG FUNCTION
#   Each stage is independently testable (see tests/test_rag_pipeline.py,
#   which tests build_context's formatting without calling any LLM), and
#   independently reusable -- e.g. a future admin "debug this retrieval"
#   tool could call retrieve() + build_context() without ever generating an
#   answer.
#
# LANGCHAIN CONCEPT: ChatGoogleGenerativeAI
#   `langchain_google_genai.ChatGoogleGenerativeAI` wraps the Gemini API
#   behind LangChain's standard chat-model interface (`.invoke(messages)`),
#   the same interface every other LangChain-supported model uses. That
#   means the query rewriter (which also calls `.invoke(messages)`) and this
#   pipeline share one Gemini client instance and one calling convention --
#   swapping to a different LangChain-supported model later would not
#   require touching prompt_templates.py or query_rewriter.py at all.
#
# STREAMING
#   generate_answer()/answer_question() accept an optional `on_token`
#   callback. When given, the final answer streams token-by-token (via the
#   LangChain chat model's `.stream()` instead of `.invoke()`) and `on_token`
#   is called with the accumulated text so far on each new piece -- see
#   generate_answer()'s docstring. Retrieval, reranking, grounding, memory,
#   and MLflow logging are unaffected either way; only how the final answer
#   text is delivered changes.
#
# CONCURRENCY
#   Multi-query retrieval's per-variant searches (rag_pipeline/multi_query.py)
#   run concurrently via a thread pool in retrieve_with_stages() -- they're
#   independent embedding + Chroma calls with no shared state, so running
#   them one after another was pure wasted wall-clock time. This is threads,
#   not asyncio: the underlying I/O (the embedding provider, ChromaDB) is
#   ordinary blocking Python, and threads already release the GIL during
#   network I/O, so a full async rewrite of the retrieval stack wouldn't buy
#   more concurrency here, just more code.
#
# INPUT / OUTPUT
#   answer_question(question) -> {"answer": str, "sources": [...],
#   "confidence": float} -- everything the Streamlit UI needs to render one
#   chat turn.
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from rag_pipeline import guardrails, retrieval_service
from rag_pipeline.conversation import conversational_reply
from rag_pipeline.llm_service import acquire_chat_slot, get_chat_model
from rag_pipeline.memory import ConversationMemory
from rag_pipeline.multi_hop import plan_next_hop
from rag_pipeline.multi_query import generate_query_variants, reciprocal_rank_fusion
from rag_pipeline.prompt_templates import NOT_FOUND_MESSAGE, build_prompt
from rag_pipeline.query_rewriter import rewrite_query
from rag_pipeline.reranker import rerank
from rag_pipeline.retrieval_service import RetrievedChunk, chunk_identity
from utils.logging_utils import get_logger
from utils.mlflow_tracking import trace_query
from utils.service_errors import classify_error, is_transient

logger = get_logger(__name__)

SERVICE_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the AI service right now (it may be rate-limited "
    "or temporarily unavailable). Please try asking again in a moment."
)
AUTH_ERROR_MESSAGE = (
    "I can't reach the AI service because its API key appears to be invalid, "
    "expired, or revoked. Please check GEMINI_API_KEY (or the active provider's "
    "credentials) in your .env file, then restart the app."
)

# Markers seen in real Gemini/Databricks auth failures (401 Unauthenticated,
# "invalid authentication credentials", a revoked/expired key). These are
# PERMANENT failures: retrying with backoff just wastes 10-20 seconds before
# failing again with the exact same error, and the generic "rate-limited or
# temporarily unavailable" message actively misleads the user into thinking
# the problem will resolve itself if they just wait -- it won't, until the
# key is fixed.
_PERMANENT_AUTH_ERROR_MARKERS = (
    "unauthenticated",
    "invalid authentication credentials",
    "permission_denied",
    "api key not valid",
    "api_key_invalid",
)


def _is_permanent_auth_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_AUTH_ERROR_MARKERS)


def _response_text(content):
    """LangChain models may return strings or typed content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block if isinstance(block, str) else block.get("text", "")
                       for block in content if isinstance(block, str) or
                       (isinstance(block, dict) and block.get("type") == "text"))
    return ""


@dataclass
class RetrievalStages:
    """The candidate ranking before AND after reranking, for one question."""

    before_reranking: List[RetrievedChunk]
    after_reranking: List[RetrievedChunk]


class RAGPipeline:
    """
    Orchestrates one full question-answering turn: rewrite -> retrieve ->
    rerank -> build context -> generate -> guardrail-check.

    One instance lives in each browser's session state. Its conversation memory
    is private to that session; model factories share the expensive clients.
    """

    def __init__(self):
        settings.validate()
        self._llm = get_chat_model()
        self.memory = ConversationMemory()
        logger.info(
            "RAGPipeline ready (llm_provider=%s, embedding_provider=%s)",
            settings.llm_provider,
            settings.embedding_provider,
        )

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """
        Retrieve the top-K most relevant chunks for `question`.

        Thin wrapper around retrieve_with_stages() that returns just the
        final, post-reranking list -- what the rest of the pipeline treats
        as "the evidence". Use retrieve_with_stages() directly when you also
        need the pre-reranking ranking (e.g. to measure what reranking
        actually changed -- see scripts/evaluate_retrieval.py).
        """
        return self.retrieve_with_stages(question).after_reranking

    def retrieve_with_stages(self, question: str) -> RetrievalStages:
        """
        Same retrieval as retrieve(), but returns BOTH the ranking before
        reranking and the final ranking after it, instead of discarding the
        pre-reranking one.

        WHY THIS EXISTS
          retrieve() used to compute the pre-reranking candidates, rerank
          them, and only ever return the reranked result -- there was no way
          to see what reranking actually changed. scripts/evaluate_retrieval.py
          uses this to report Recall/Precision/NDCG/MRR BEFORE and AFTER
          reranking side by side, so you can see reranking's real effect on
          retrieval quality instead of just trusting that it helps.

        Runs vector (+ optional hybrid) search via retrieval_service --
        optionally against several LLM-generated paraphrasings of `question`
        (see rag_pipeline/multi_query.py), fused via reciprocal rank fusion
        when ENABLE_MULTI_QUERY is set. `before_reranking` is that full
        candidate pool (deliberately NOT cut to top_k when reranking is
        enabled, since retrieval_service over-fetches for reranking to work
        with -- truncating first would hide exactly the "reranking promoted
        a candidate from rank 8 to rank 2" cases this method exists to show).

        The per-variant searches (when multi-query is on) are independent of
        each other -- each is its own embedding call + Chroma query with no
        shared state -- so they run concurrently via a thread pool instead of
        one after another. This is threads, not asyncio: the underlying work
        (the embedding provider, ChromaDB's client) is ordinary blocking I/O,
        not async-native, and threads release the GIL during that I/O just
        like async would, for a fraction of the rewrite. utils/rate_limiter.py
        (shared by the embedding provider) is already lock-protected, so
        concurrent callers pace correctly against the same quota rather than
        each under-counting the others' requests.

        The score threshold is enforced TWICE: once inside
        retrieval_service.retrieve() on the vector/hybrid score, and again
        here on the final score actually shown to the user. Reranking can
        assign a candidate a very different (and more accurate) score than
        the first stage did -- a chunk that barely cleared the first
        threshold can still get a near-zero cross-encoder score, and
        without this second check it would still be displayed as a
        "source" despite being effectively irrelevant. This is also why
        after_reranking can be shorter than top_k, or empty: a fixed source
        count that pads out with weak matches is exactly what a real
        relevance threshold is supposed to prevent.
        """
        if settings.enable_multi_query:
            variants = generate_query_variants(question, self._llm, settings.multi_query_variants)
            with ThreadPoolExecutor(max_workers=len(variants)) as executor:
                variant_results = list(executor.map(retrieval_service.retrieve, variants))
            before_reranking = reciprocal_rank_fusion(variant_results)
        else:
            before_reranking = retrieval_service.retrieve(question)

        if not before_reranking:
            return RetrievalStages(before_reranking=[], after_reranking=[])

        if settings.enable_reranking:
            candidate_dicts = [
                {
                    "chunk_text": c.chunk_text,
                    "score": c.score,
                    "document_name": c.document_name,
                    "document_type": c.document_type,
                    "page_number": c.page_number,
                    "chapter_title": c.chapter_title,
                    "section_title": c.section_title,
                }
                for c in before_reranking
            ]
            reranked_dicts = rerank(question, candidate_dicts)
            after_reranking = [
                RetrievedChunk(
                    chunk_text=d["chunk_text"],
                    score=d.get("rerank_score", d["score"]),
                    document_name=d["document_name"],
                    document_type=d["document_type"],
                    page_number=d["page_number"],
                    chapter_title=d.get("chapter_title", ""),
                    section_title=d.get("section_title", ""),
                )
                for d in reranked_dicts
                if d.get("rerank_score", d["score"]) >= settings.score_threshold
            ][: settings.top_k]
        else:
            after_reranking = before_reranking[: settings.top_k]

        return RetrievalStages(before_reranking=before_reranking, after_reranking=after_reranking)

    def _run_multi_hop(
        self, question: str, chunks: List[RetrievedChunk], context: str
    ) -> "tuple[List[RetrievedChunk], str]":
        """
        Let the model ask itself up to (settings.max_hops - 1) follow-up
        search queries when the current context doesn't fully answer
        `question` (see rag_pipeline/multi_hop.py), merging each hop's
        chunks into the running set and rebuilding the context each time.

        Bounded by settings.max_hops -- a single sufficiency-check failure
        or an already-sufficient context stops the loop immediately, so the
        common case costs nothing beyond the one extra check.
        """
        hops_done = 1
        seen_identities = {chunk_identity(c) for c in chunks}

        while hops_done < settings.max_hops:
            follow_up_query = plan_next_hop(question, context, self._llm)
            if not follow_up_query:
                break

            new_chunks = [c for c in self.retrieve(follow_up_query) if chunk_identity(c) not in seen_identities]
            if not new_chunks:
                # Nothing new came back -- another hop would just repeat
                # this same check against unchanged context.
                break

            chunks = chunks + new_chunks
            seen_identities.update(chunk_identity(c) for c in new_chunks)
            context = self.build_context(chunks)
            hops_done += 1

        return chunks, context

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into the citation-ready text block the LLM
        sees in its prompt.

        Each chunk is labeled with a bracketed source tag
        ([Source: <document>, page <N>], with chapter/section appended when
        the source document has that structure) directly above its text, so
        the model has an unambiguous, ready-to-copy citation for every piece
        of context it might use in its answer.
        """
        if not chunks:
            return "(No relevant context was found in the available documents.)"

        blocks = []
        for chunk in chunks:
            citation = f"{chunk.document_name}, page {chunk.page_number}"
            if chunk.section_title:
                citation += f" ({chunk.section_title})"
            blocks.append(f"[Source: {citation}]\n{chunk.chunk_text}")
        return "\n\n---\n\n".join(blocks)

    @retry(
        # Chat APIs (Gemini's free tier especially -- 5 requests/minute) can
        # return transient 429/503 errors under normal interactive use, not
        # just under load. A short, bounded retry smooths over a single
        # rate-limit blip without making the user wait too long if the
        # service is genuinely down.
        #
        # retry_error_callback below re-raises immediately (no backoff at
        # all) when the underlying error is a permanent auth failure (a
        # dead/invalid API key): retrying that with backoff just delays the
        # same guaranteed failure by 10-20 seconds for nothing.
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=lambda retry_state: (
            retry_state.outcome.failed
            and is_transient(retry_state.outcome.exception())
        ),
        reraise=True,
    )
    def _invoke_llm(self, messages: List) -> str:
        acquire_chat_slot()
        response = self._llm.invoke(messages)
        return _response_text(response.content)

    @retry(
        # Same policy as _invoke_llm() above -- this decorator wraps ONE
        # attempt at streaming the full response. If a failure happens after
        # some chunks were already sent to `on_token`, the retried attempt
        # starts its own accumulation from empty and calls `on_token` with
        # that fresh (shorter) text -- the caller's UI naturally re-renders
        # from scratch rather than appending a stale partial answer to a new
        # one, since each call passes the full accumulated-so-far text, not
        # a delta.
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=lambda retry_state: (
            retry_state.outcome.failed
            and is_transient(retry_state.outcome.exception())
        ),
        reraise=True,
    )
    def _stream_llm(self, messages: List, on_token: Callable[[str], None]) -> str:
        acquire_chat_slot()
        accumulated = ""
        for chunk in self._llm.stream(messages):
            piece = _response_text(chunk.content)
            if piece:
                accumulated += piece
                on_token(accumulated)
        if not accumulated.strip():
            raise ValueError("The model returned an empty response.")
        return accumulated.strip()

    def generate_answer(self, context, question, on_token=None):
        messages = build_prompt(context, question, self.memory.get_history())
        self._generation_issue = None
        try:
            if on_token is not None and settings.enable_streaming:
                try:
                    answer = self._stream_llm(messages, on_token)
                except Exception as exc:
                    if classify_error(exc).code in {"authentication", "permission", "model", "quota"}:
                        raise
                    logger.warning("Streaming failed; trying a normal response.", exc_info=True)
                    # A failed stream may already have rendered a partial response.
                    # The final full answer replaces it in the UI.
                    answer = self._invoke_llm(messages)
            else:
                answer = self._invoke_llm(messages)
            if not answer.strip():
                raise ValueError("The model returned an empty response.")
            return answer.strip()
        except Exception as exc:
            logger.exception("Chat model call failed")
            self._generation_issue = classify_error(exc)
            return self._generation_issue.message

    def answer_question(self, question, on_token=None, on_status=None):
        """Run a turn, returning explicit errors without polluting chat memory."""
        # Exact social phrases only: mixed greetings + policy questions still use RAG.
        # Do not store social turns in retrieval memory or consume provider quota.
        reply = conversational_reply(question)
        if reply is not None:
            return {"answer": reply, "sources": [], "confidence": None,
                    "grounded": None, "status": "conversational", "warnings": []}
        allowed, reason = guardrails.check_input(question)
        if not allowed:
            return {"answer": reason, "sources": [], "confidence": 0.0,
                    "grounded": True, "status": "rejected", "warnings": []}
        notify = on_status or (lambda message: None)
        warnings = []
        sources = []
        stage = "retrieval"
        model = settings.gemini_chat_model if settings.llm_provider == "gemini" else settings.databricks_llm_endpoint
        try:
            with trace_query(question, top_k=settings.top_k, chat_model=model) as run_data:
                notify("Understanding your question (API quota may require a short wait)...")
                standalone = rewrite_query(question, self.memory.get_history(), self._llm)
                notify("Searching the indexed documents and ranking evidence...")
                chunks = self.retrieve(standalone)
                if not chunks:
                    answer = NOT_FOUND_MESSAGE
                    self.memory.add_turn(question, answer)
                    run_data["answer"] = answer
                    return {"answer": answer, "sources": [], "confidence": 0.0,
                            "grounded": True, "status": "not_found", "warnings": []}
                context = self.build_context(chunks)
                if settings.enable_multi_hop:
                    notify("Checking whether more evidence is needed...")
                    try:
                        chunks, context = self._run_multi_hop(standalone, chunks, context)
                    except Exception:
                        logger.warning("Follow-up search failed; keeping the original evidence.", exc_info=True)
                        warnings.append("The additional evidence search failed; this answer uses the first search results.")
                sources = [{"document_name": c.document_name, "document_type": c.document_type,
                            "page_number": c.page_number, "score": c.score,
                            "chapter_title": c.chapter_title, "section_title": c.section_title,
                            "excerpt": c.chunk_text} for c in chunks]
                run_data["retrieved_chunks"] = sources
                stage = "generation"
                notify("Evidence found. Waiting for the AI service to answer (quota limits may delay this step)...")
                answer = self.generate_answer(context, standalone, on_token=on_token)
                run_data["answer"] = answer
                issue = getattr(self, "_generation_issue", None)
                if issue is not None:
                    return {"answer": answer, "sources": sources, "confidence": None,
                            "grounded": None, "status": "error", "error_code": issue.code,
                            "stage": stage, "warnings": warnings}
                grounded = guardrails.check_grounding(answer, [c.chunk_text for c in chunks])
                self.memory.add_turn(question, answer)
                return {"answer": answer, "sources": sources,
                        "confidence": round(sum(c.score for c in chunks) / len(chunks), 3),
                        "grounded": grounded, "status": "ok", "warnings": warnings}
        except Exception as exc:
            logger.exception("Chat turn failed during %s", stage)
            issue = classify_error(exc)
            return {"answer": issue.message, "sources": sources, "confidence": None,
                    "grounded": None, "status": "error", "error_code": issue.code,
                    "stage": stage, "warnings": warnings}
