"""Session-owned conversation state; models remain shared by their factories."""
from rag_pipeline.rag_pipeline import RAGPipeline


def get_session_pipeline(state):
    if "rag_pipeline" not in state:
        state["rag_pipeline"] = RAGPipeline()
    return state["rag_pipeline"]
