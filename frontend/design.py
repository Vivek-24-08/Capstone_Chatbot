"""Static visual components; no remote assets or extra runtime dependencies."""
from pathlib import Path

import streamlit as st


def apply_design():
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header(in_conversation):
    st.markdown('''<div class="cg-topbar"><span>YOUR BENEFITS COMPANION</span>
        <span class="cg-topbar-right">CareGuide <span class="cg-dot">●</span> Document-based answers</span></div>''', unsafe_allow_html=True)
    if in_conversation:
        st.markdown('''<div class="cg-conversation-title"><span class="cg-kicker">LET'S MAKE IT CLEAR</span>
            <h1>Your coverage, in conversation.</h1>
            <p>Ask a follow-up, explore the details, and review the supporting passages.</p></div>''', unsafe_allow_html=True)
        return
    st.markdown('''<section class="cg-hero">
      <div class="cg-hero-copy">
        <div class="cg-pill">HEALTHCARE INSURANCE, SIMPLIFIED</div>
        <h1>Less policy jargon.<br><span>More peace of mind.</span></h1>
        <p>A clearer way to understand your benefits. Ask a question and explore
        what your plan documents actually say.</p>
        <div class="cg-hero-footer"><span>01 &nbsp; Ask naturally</span><span>02 &nbsp; Explore the evidence</span></div>
      </div>
      <div class="cg-illustration" aria-hidden="true">
        <div class="cg-orbit"></div>
        <div class="cg-paper"><div class="cg-paper-icon">+</div><strong>Your plan.<br>Made clearer.</strong>
          <div class="cg-line"></div><div class="cg-line short"></div><div class="cg-line"></div>
          <div class="cg-paper-label">COVERAGE &nbsp; / &nbsp; BENEFITS</div></div>
        <div class="cg-floating">↗ &nbsp; From questions to clarity</div>
      </div>
    </section>''', unsafe_allow_html=True)


def render_status(passage_count):
    # Count is numeric; no document/user/model text is interpolated as HTML.
    st.markdown(f'''<div class="cg-status-row">
      <div><span class="cg-status-icon">▤</span><span><strong>{int(passage_count):,} searchable passages</strong><small>In your document index</small></span></div>
      <div><span class="cg-status-icon">↗</span><span><strong>Evidence alongside answers</strong><small>Review the original passages</small></span></div>
      <div><span class="cg-status-icon">↔</span><span><strong>Room for follow-ups</strong><small>Build on your conversation</small></span></div>
    </div>''', unsafe_allow_html=True)


def conversation_export(messages):
    sections = ["# CareGuide conversation", "Review cited plan documents and confirm coverage decisions with your insurer."]
    for message in messages:
        sections.append(f"## {'You' if message['role'] == 'user' else 'CareGuide'}\n\n{message['content']}")
        for source in message.get("sources", []):
            sections.append(f"Source: {source['document_name']}, page {source['page_number']}\n\n{source.get('excerpt', '')}")
    return "\n\n".join(sections)
