"""Streamlit chat UI. Launch remains: streamlit run frontend/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

st.set_page_config(page_title="CareGuide | Insurance Assistant", page_icon="🩺", layout="wide")
from frontend.design import apply_design, render_header, render_status, conversation_export

apply_design()
try:
    from config.settings import settings
    from frontend.session import get_session_pipeline
    from scripts.ingest import run_ingestion
    from utils.feedback_logger import log_feedback
    from utils.service_errors import classify_error
    from vector_store import chroma_manager
except (ValueError, ImportError):
    st.error("Setup could not be loaded. Check numeric values in .env and install the dependencies with pip install -r requirements.txt. See the server log for details.")
    st.stop()

def load_pipeline():
    return get_session_pipeline(st.session_state)

@st.cache_resource(show_spinner=False)
def ensure_documents_ingested():
    return run_ingestion()

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="cg-brand"><span class="cg-brand-mark">+</span>CareGuide</div><div class="cg-brand-sub">Clarity for your coverage.</div>', unsafe_allow_html=True)
        if st.button("＋ New conversation", type="primary", use_container_width=True):
            st.session_state.messages = []
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.memory.clear()
            st.rerun()
        messages = st.session_state.get("messages", [])
        if messages:
            st.download_button("↓ Save conversation", conversation_export(messages),
                               file_name="careguide-conversation.md", mime="text/markdown",
                               use_container_width=True)
        st.caption("Start fresh anytime. A new conversation clears this session's chat history.")
        st.divider()
        st.markdown("**Your knowledge base**")
        with st.container(border=True):
            try:
                count = chroma_manager.count()
                st.write(f"▤  {count:,} indexed passages")
                st.caption("Available for document search" if count else "Add documents to get started")
            except Exception:
                st.caption("Index is not ready yet.")
        st.markdown("**A little guidance**")
        st.caption("Ask one clear question at a time. Mention your plan or benefit when possible, then explore the cited evidence.")
        with st.expander("Setup and connection checks"):
            st.caption(f"Chat provider: {settings.llm_provider}")
            st.caption(f"Embeddings: {settings.embedding_provider}")
            st.caption("Ingestion and chat use separate services. An indexed document does not prove that the chat endpoint is reachable.")
            if st.button("Refresh document index"):
                ensure_documents_ingested.clear()
                st.rerun()
            st.caption("The connection check makes small embedding and chat requests; provider usage may apply.")
            if st.button("Check AI connection"):
                from scripts.doctor import run_checks
                with st.spinner("Checking configured services..."):
                    try:
                        checks = run_checks(probe=True)
                        for item in checks:
                            (st.success if item["ok"] else st.error)(f"{item['check']}: {item['detail']}")
                    except Exception as exc:
                        st.error(classify_error(exc).message)
        st.divider()
        with st.expander("About your conversation"):
            st.caption("Chat memory is separate for each browser session. Questions and relevant passages may be sent to the configured AI provider. Feedback and tracing may be stored.")
            st.caption("Document availability does not confirm AI service availability. Use the connection check if answers fail.")
        st.caption("Use this assistant to understand documents. Confirm coverage decisions with your insurer.")

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"▤  Sources & evidence · {len(sources)} passages"):
        for number, source in enumerate(sources, 1):
            with st.container(border=True):
                st.write(f"[{number}] {source['document_name']}")
                st.caption(f"Page {source['page_number']} · Source relevance: {source['score']:.2f}")
                if source.get("section_title"):
                    st.caption(source["section_title"])
                if source.get("excerpt"):
                    # Markdown is escaped so extracted document text cannot inject links or formatting.
                    import re
                    excerpt = re.sub(r"([\\`*_{}\[\]()<>#+.!|~-])", r"\\\1", source["excerpt"])
                    st.markdown(excerpt)

def render_assistant_message(index, message, skip_content=False):
    if not skip_content:
        if message.get("status") == "error":
            st.error(message["content"])
        else:
            st.markdown(message["content"])
    render_sources(message.get("sources", []))
    if message.get("status") == "not_found":
        st.info("Try naming the benefit or plan, or rephrasing your question. The needed document may not be uploaded. For details outside these documents, check with your insurer; for unrelated topics, use a general-purpose assistant.")
    if message.get("confidence") is not None and message.get("sources"):
        st.caption(f"Retrieved-source relevance: {message['confidence']:.0%} · This is not an answer-accuracy score.")
    if message.get("grounded") is False:
        st.warning("The support check flagged this answer. Verify the cited passages before relying on it.")
    for warning in message.get("warnings", []):
        st.warning(warning)
    if message.get("status") == "error":
        st.caption("If evidence is shown above, document search worked but answer generation did not. Use the connection check in the sidebar.")
        return
    st.caption("Was this useful?")
    left, right = st.columns(2)
    for column, label, rating in ((left, "👍 Helpful", "up"), (right, "👎 Not helpful", "down")):
        with column:
            if st.button(label, key=f"{rating}_{index}"):
                saved = log_feedback(message.get("question", ""), message["content"], message.get("sources", []), rating=rating)
                if saved:
                    st.toast("Feedback saved.")
                else:
                    st.warning("Feedback could not be saved. Your answer is unaffected.")

def main():
    render_header(bool(st.session_state.get("messages")))
    try:
        settings.validate()
    except ValueError as exc:
        st.error(str(exc))
        st.info("Update .env, then restart Streamlit. Never paste an API key into chat.")
        return
    render_sidebar()
    try:
        with st.spinner("Checking the document index..."):
            ensure_documents_ingested()
    except Exception as exc:
        # A failed refresh must not make a healthy, compatible previous index unusable.
        try:
            chroma_manager.assert_compatible_index()
            usable = chroma_manager.count() > 0
        except Exception:
            usable = False
        if not usable:
            st.error("Document indexing failed. " + classify_error(exc).message)
            st.info("Run python -m scripts.ingest in the terminal for details, then use Refresh document index.")
            return
        st.warning("Document refresh failed. Answers will use the previously published index until refresh succeeds.")
    try:
        chroma_manager.assert_compatible_index()
        if chroma_manager.count() == 0:
            st.info("No documents are indexed. Add text-based PDFs to the configured document folder and refresh the index.")
            return
        pipeline = load_pipeline()
    except Exception as exc:
        st.error(classify_error(exc).message)
        return
    if "messages" not in st.session_state:
        st.session_state.messages = []
    render_status(chroma_manager.count())
    suggested_question = None
    if not st.session_state.messages:
        st.markdown('<div class="cg-kicker">A GOOD PLACE TO START</div>', unsafe_allow_html=True)
        st.subheader("What would you like to explore?")
        st.caption("Pick a topic below, or ask a question in your own words.")
        prompts = [
            ("01 / COSTS", "Understand your costs", "Deductibles, copayments, and what you pay out of pocket.", "What does my plan say about deductibles and out-of-pocket costs?"),
            ("02 / BENEFITS", "Know what's covered", "Explore the services and benefits described in your plan.", "What benefits and coverage are described in my plan?"),
            ("03 / PLAN RULES", "See the fine print", "Understand exclusions, coverage limits, and conditions.", "What exclusions and coverage limits should I know about?"),
        ]
        for column, (eyebrow, title, description, prompt) in zip(st.columns(3), prompts):
            with column, st.container(border=True):
                st.markdown(f'<div class="cg-topic"><small>{eyebrow}</small><strong>{title}</strong><p>{description}</p></div>', unsafe_allow_html=True)
                if st.button("Explore topic →", key=eyebrow, use_container_width=True):
                    suggested_question = prompt
        st.caption("Answers are based on your indexed documents. If a detail isn't supported, CareGuide will say so.")
    else:
        st.caption("YOUR CONVERSATION · Questions and supporting evidence in one place")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="🩺" if message["role"] == "assistant" else None):
            if message["role"] == "assistant":
                render_assistant_message(index, message)
            else:
                st.markdown(message["content"])
    question = st.chat_input("Ask about your coverage, deductibles, or benefits...") or suggested_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🩺"):
            placeholder = st.empty()
            try:
                result = pipeline.answer_question(
                    question,
                    on_token=lambda text: placeholder.markdown(text + "▌"),
                    on_status=lambda text: placeholder.info(text),
                )
            except Exception as exc:
                # Last UI boundary: no unfinished placeholder or raw traceback.
                result = {"answer": classify_error(exc).message, "sources": [], "confidence": None,
                          "grounded": None, "status": "error", "warnings": []}
            if result.get("status") == "error":
                placeholder.error(result["answer"])
            else:
                placeholder.markdown(result["answer"])
            message = dict(result, role="assistant", content=result["answer"], question=question)
            st.session_state.messages.append(message)
            st.rerun()

if __name__ == "__main__":
    main()
