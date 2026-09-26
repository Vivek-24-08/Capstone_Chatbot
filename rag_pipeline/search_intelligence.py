"""Explainable insurance-query routing for hybrid retrieval."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SearchProfile:
    intent: str
    keyword_query: str
    vector_weight: float
    keyword_weight: float


_ROUTES = (
    ("costs", ("deductible", "copay", "coinsurance", "premium", "cost", "pay", "out of pocket"),
     ("member responsibility", "allowed amount", "cost sharing"), .55, .45),
    ("pharmacy", ("drug", "prescription", "pharmacy", "formulary", "medication", "generic", "brand"),
     ("preferred drug", "specialty drug", "prior authorization"), .58, .42),
    ("claims", ("claim", "reimbursement", "submit", "denied", "explanation of benefits", "eob"),
     ("claim form", "filing deadline", "appeal"), .58, .42),
    ("appeals", ("appeal", "grievance", "complaint", "adverse benefit", "denial"),
     ("review request", "filing deadline", "external review"), .55, .45),
    ("providers", ("provider", "doctor", "hospital", "network", "specialist", "out of network"),
     ("in network", "participating provider", "referral"), .62, .38),
    ("exclusions", ("exclude", "exclusion", "not covered", "limitation", "limit", "maximum"),
     ("coverage limitation", "benefit maximum", "non covered service"), .58, .42),
    ("coverage", ("cover", "coverage", "benefit", "eligible", "service", "treatment"),
     ("covered service", "medical necessity", "eligibility"), .68, .32),
)


def build_search_profile(question: str, enabled: bool = True) -> SearchProfile:
    """Detect an intent and tune only the lexical branch; vectors use the original question."""
    clean = " ".join(question.strip().split())
    if not enabled:
        return SearchProfile("general", clean, .70, .30)
    lowered = clean.lower()
    for intent, terms, expansions, vector_weight, keyword_weight in _ROUTES:
        if any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in terms):
            additions = [term for term in expansions if term not in lowered]
            return SearchProfile(intent, " ".join([clean, *additions]), vector_weight, keyword_weight)
    return SearchProfile("general", clean, .70, .30)
