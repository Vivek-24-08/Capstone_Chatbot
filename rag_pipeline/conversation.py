"""Conservative local routing for social turns; never intercept mixed questions."""
import re


def conversational_reply(question: str):
    text = re.sub(r"[^\w\s']", " ", question.lower().replace("’", "'"))
    text = " ".join(text.split())
    greeting = r"(?:hi|hello|hey|good morning|good afternoon|good evening)"
    wellbeing = r"(?:how are you(?: doing)?|how's it going|how do you do)"
    if re.fullmatch(rf"(?:{greeting}(?: there)?(?: {wellbeing})?|{wellbeing})", text):
        return (
            "Hello! I'm CareGuide, ready to help you understand your healthcare insurance documents. "
            "You can ask about coverage, deductibles, exclusions, or claims. What would you like to explore?"
        )
    if text in {"thanks", "thank you", "thanks a lot", "thank you very much", "great thanks"}:
        return "You're welcome! Feel free to ask another question about your plan."
    if text in {"bye", "goodbye", "see you", "good night"}:
        return "Goodbye! Come back whenever you'd like to explore your plan documents."
    if text in {"help", "who are you", "what can you do", "what can i ask", "how can you help me"}:
        return (
            "I'm CareGuide, your document-based healthcare insurance assistant. "
            "I can explain the coverage, costs, exclusions, and claims rules described in your uploaded documents, "
            "and show the supporting passages. Try: 'What is my deductible?' or 'What exclusions apply?' "
            "If a detail isn't supported by the documents, I'll say so. For unrelated topics, use a general-purpose assistant."
        )
    return None
