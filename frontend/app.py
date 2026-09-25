"""Streamlit chat UI. Launch remains: streamlit run frontend/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

st.set_page_config(page_title="CareGuide | Insurance Assistant", page_icon="🩺", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
.care-hero {background: linear-gradient(120deg,#102d40,#12675f); color: white;
 padding: 2rem 2.2rem; border-radius: 22px; margin-bottom: 1.4rem;}
.care-eyebrow {font-size: .75rem; letter-spacing: .16em; font-weight: 700; color: #9fe1d4;}
.care-hero h1 {color: white; font-size: clamp(1.8rem,4vw,2.7rem); padding: .5rem 0; line-height: 1.2;}
.care-hero p {color: #d8ebe9; max-width: 650px; margin-bottom: 0; line-height: 1.7;}
.care-brand {font-size: 1.65rem; font-weight: 750; margin-bottom: .2rem;}
[data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.18); border-radius: 16px; padding: 1.2rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.2); border-radius: 14px; padding: 1rem;}
.stButton > button {border-radius: 10px; min-height: 2.8rem;}
@media (max-width: 640px) {.care-hero {padding: 1.4rem;} .block-container {padding-top: 1rem;}}
</style>
""", unsafe_allow_html=True)
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
        st.markdown('<div class="care-brand">🩺 CareGuide</div>', unsafe_allow_html=True)
        st.caption("YOUR POLICY, MADE CLEARER")
        st.divider()
        st.write("Explore your healthcare benefits with answers supported by your plan documents.")
        st.subheader("Knowledge base")
        try:
            st.write(f"{chroma_manager.count()} indexed passages")
        except Exception:
            st.caption("Index is not ready yet.")
        if st.button("＋ New conversation", use_container_width=True):
            st.session_state.messages = []
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.memory.clear()
            st.rerun()
        st.divider()
        st.markdown("**How it works**")
        st.caption("1. Ask a question about your plan.\n\n2. Review the answer and cited passages.\n\n3. Ask a follow-up to explore the details.")
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
        st.caption("Use this assistant to understand documents. Confirm coverage decisions with your insurer.")

def render_sources(sources):
    if not sources:
        return
    with st.expander(f"View evidence ({len(sources)} passages)"):
        for number, source in enumerate(sources, 1):
            with st.container(border=True):
                st.write(f"[{number}] {source['document_name']}")
                st.caption(f"Page {source['page_number']} · Source relevance: {source['score']:.2f}")
                if source.get("section_title"):
                    st.caption(source["section_title"])
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
    st.markdown('''<div class="care-hero">
    <div class="care-eyebrow">CAREGUIDE · HEALTHCARE INSURANCE ASSISTANT</div>
    <h1>Understand your coverage.<br>Find the details that matter.</h1>
    <p>Ask about benefits, deductibles, and plan rules. Explore answers alongside
    the passages from your policy that support them.</p></div>''', unsafe_allow_html=True)
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
    overview = st.columns(3)
    overview[0].metric("Indexed passages", f"{chroma_manager.count():,}", help="Number of indexed document passages, not documents.")
    overview[1].metric("Evidence", "Source passages")
    overview[2].metric("Conversation", "Session memory", help="Conversation memory is separate for each browser session. Questions and retrieved passages may be sent to your configured AI provider; feedback and tracing may be stored.")
    st.caption("Indexed passages are ready for search. AI service availability depends on your configured provider and quota.")
    suggested_question = None
    if not st.session_state.messages:
        st.subheader("What would you like to understand?")
        st.caption("Choose a starting point, or write your own question below.")
        prompts = [
            ("💳 Costs & deductibles", "What does my plan say about deductibles and out-of-pocket costs?"),
            ("🩺 Benefits & coverage", "What benefits and coverage are described in my plan?"),
            ("📋 Exclusions & limits", "What exclusions and coverage limits should I know about?"),
        ]
        for column, (label, prompt) in zip(st.columns(3), prompts):
            with column:
                if st.button(label, use_container_width=True):
                    suggested_question = prompt
        st.divider()
    else:
        st.subheader("Your conversation")
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(index, message)
            else:
                st.markdown(message["content"])
    question = st.chat_input("Ask about your coverage, deductibles, or benefits...") or suggested_question
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
