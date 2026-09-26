import pytest

from rag_pipeline.search_intelligence import build_search_profile


@pytest.mark.parametrize("question,intent", [
    ("What is my deductible?", "costs"),
    ("Is this prescription on the formulary?", "pharmacy"),
    ("How do I appeal a denial?", "appeals"),
    ("Is my specialist in network?", "providers"),
    ("Which services are not covered?", "exclusions"),
    ("Tell me about an unrelated document detail", "general"),
])
def test_routes_insurance_intents(question, intent):
    profile = build_search_profile(question)
    assert profile.intent == intent
    assert profile.vector_weight + profile.keyword_weight == pytest.approx(1)


def test_cost_search_adds_domain_terms_and_increases_keyword_weight():
    profile = build_search_profile("What is my annual deductible?")
    assert "member responsibility" in profile.keyword_query
    assert profile.keyword_weight > .30


def test_disabled_intelligence_preserves_query_and_standard_weights():
    profile = build_search_profile("What is my deductible?", enabled=False)
    assert profile.intent == "general"
    assert profile.keyword_query == "What is my deductible?"
    assert profile.vector_weight == .70
    assert profile.keyword_weight == .30
