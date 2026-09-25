"""Streamlit chat UI. Launch remains: streamlit run frontend/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

st.set_page_config(page_title="Healthcare Insurance Assistant", page_icon="🩺", layout="centered")
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
        st.header("Healthcare Insurance Assistant")
        st.caption("Answers grounded in your plan documents.")
        st.subheader("Knowledge base")
        try:
            st.write(f"{chroma_manager.count()} indexed passages")
        except Exception:
            st.caption("Index is not ready yet.")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.memory.clear()
            st.rerun()
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

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"View evidence ({len(sources)} passages)"):
        for source in sources:
            st.markdown(f"**{source['document_name']}**, page {source['page_number']}")
            if source.get("section_title"):
                st.caption(source["section_title"])
            st.caption(f"Source relevance: {source['score']:.2f}")
            if source.get("excerpt"):
                st.text(source["excerpt"])

def render_assistant_message(index, message, skip_content=False):
    if not skip_content:
        if message.get("status") == "error":
            st.error(message["content"])
        else:
            st.markdown(message["content"])
    render_sources(message.get("sources", []))
    if message.get("confidence") is not None and message.get("sources"):
        st.caption(f"Retrieved-source relevance: {message['confidence']:.0%} · This is not an answer-accuracy score.")
    if message.get("grounded") is False:
        st.warning("The support check flagged this answer. Verify the cited passages before relying on it.")
    for warning in message.get("warnings", []):
        st.warning(warning)
    if message.get("status") == "error":
        st.caption("If evidence is shown above, document search worked but answer generation did not. Use the connection check in the sidebar.")
        return
    left, right = st.columns(2)
    for column, label, rating in ((left, "Helpful", "up"), (right, "Not helpful", "down")):
        with column:
            if st.button(label, key=f"{rating}_{index}"):
                saved = log_feedback(message.get("question", ""), message["content"], message.get("sources", []), rating=rating)
                if saved:
                    st.toast("Feedback saved.")
                else:
                    st.warning("Feedback could not be saved. Your answer is unaffected.")

def main():
    st.title("🩺 Healthcare Insurance Assistant")
    st.caption("Ask about benefits, coverage, and plan rules. Review the cited evidence alongside each answer.")
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
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(index, message)
            else:
                st.markdown(message["content"])
    question = st.chat_input("Ask about your coverage, deductibles, or benefits...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
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
            render_assistant_message(len(st.session_state.messages) - 1, message, skip_content=True)

if __name__ == "__main__":
    main()
