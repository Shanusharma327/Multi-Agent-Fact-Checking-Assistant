"""Streamlit dashboard for the Multi-Agent Fact-Checking Assistant.

Left column: the grounded three-agent pipeline, traced stage by stage.
Right column: the ungrounded single-LLM baseline, for direct comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from fact_checker import (
    Adjudication,
    AuditReport,
    BaselineResult,
    MultiAgentResult,
    ScoutReport,
    Settings,
)
from fact_checker.config import ollama_status
from fact_checker.llm import AgentError, ConfigurationError
from fact_checker.pipeline import FactCheckPipeline

logging.basicConfig(level=logging.INFO)
# The search stack logs every engine attempt (including ones it internally
# retries past) at INFO, which buries our own messages in the console.
for _noisy in ("ddgs", "primp", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

st.set_page_config(
    page_title="Multi-Agent Fact-Checking Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

VERDICT_STYLE: dict[str, dict[str, str]] = {
    "SUPPORTS": {"emoji": "🟢", "color": "#1f9d55", "label": "SUPPORTS"},
    "REFUTES": {"emoji": "🔴", "color": "#e5484d", "label": "REFUTES"},
    "NOT_ENOUGH_INFO": {"emoji": "🟡", "color": "#d99a00", "label": "NOT ENOUGH INFO"},
}

EXAMPLE_CLAIMS: list[tuple[str, str, str]] = [
    (
        "🕰️ Temporal twist",
        "A claim that was true once and quietly stopped being true.",
        "The Eiffel Tower was the tallest man-made structure in the world when "
        "it was completed, and it still holds that title today.",
    ),
    (
        "🧪 Urban myth",
        "A widely repeated claim that evidence refutes.",
        "Humans only use 10 percent of their brains.",
    ),
    (
        "📈 Recent dynamic fact",
        "A fast-moving fact where parametric memory goes stale.",
        "The current Secretary-General of the United Nations is Antonio Guterres.",
    ),
]

CSS = """
<style>
/* Hide Streamlit's own chrome: the Deploy button, the three-dot main menu,
   and the footer. Belt-and-braces alongside client.toolbarMode="minimal" in
   .streamlit/config.toml, which some versions apply only to the menu. */
#MainMenu, [data-testid="stMainMenu"] { visibility: hidden; display: none; }
.stDeployButton, [data-testid="stDeployButton"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
footer { visibility: hidden; display: none; }
[data-testid="stDecoration"] { display: none; }
.verdict-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
}
.verdict-card.accent { border-left-width: 6px; border-left-style: solid; }
.verdict-badge {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.92rem;
    letter-spacing: 0.04em;
    color: #ffffff;
}
.card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    opacity: 0.65;
    margin-bottom: 0.55rem;
}
.nli-row {
    border-left: 4px solid;
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.035);
    font-size: 0.9rem;
}
.nli-id {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 700;
}
.source-chip {
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    padding: 0.16rem 0.55rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
    border-radius: 6px;
    background: rgba(108,142,245,0.18);
    border: 1px solid rgba(108,142,245,0.40);
}
.muted { opacity: 0.6; font-size: 0.85rem; }
</style>
"""


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state() -> None:
    """Seed every session_state key the app reads, so reruns are safe."""
    defaults: dict[str, object] = {
        "claim_text": "",
        "multi_result": None,
        "baseline_result": None,
        "run_error": None,
        "baseline_error": None,
        "just_ran": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_example(text: str) -> None:
    """Populate the claim box from a quick-load button and clear stale output."""
    st.session_state.claim_text = text
    st.session_state.multi_result = None
    st.session_state.baseline_result = None
    st.session_state.run_error = None
    st.session_state.baseline_error = None


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def style_for(verdict: str) -> dict[str, str]:
    """Look up badge styling, defaulting to the neutral label."""
    return VERDICT_STYLE.get(verdict, VERDICT_STYLE["NOT_ENOUGH_INFO"])


def confidence_words(confidence: float) -> str:
    """Describe a confidence score in plain language."""
    if confidence >= 0.85:
        return "Very high"
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.50:
        return "Moderate"
    if confidence >= 0.25:
        return "Low"
    return "Very low"


def render_verdict_card(
    adjudication: Adjudication,
    *,
    title: str,
    scout: Optional[ScoutReport] = None,
) -> None:
    """Render the verdict card with badge, confidence metrics and citations."""
    style = style_for(adjudication.final_verdict)
    st.markdown(
        f"""
        <div class="verdict-card accent" style="border-left-color:{style['color']}">
            <div class="card-title">{title}</div>
            <span class="verdict-badge" style="background:{style['color']}">
                {style['emoji']}&nbsp;{style['label']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    left.metric("Confidence", f"{adjudication.confidence:.0%}")
    right.metric("Calibration", confidence_words(adjudication.confidence))
    st.progress(min(1.0, max(0.0, adjudication.confidence)))

    st.markdown("**Reasoning**")
    st.write(adjudication.detailed_reasoning)

    st.markdown("**Cited sources**")
    if not adjudication.cited_sources:
        st.markdown(
            '<span class="muted">No sources cited &mdash; this verdict rests on '
            "internal model knowledge alone.</span>",
            unsafe_allow_html=True,
        )
        return

    lookup = scout.by_id() if scout else {}
    for source_id in adjudication.cited_sources:
        item = lookup.get(source_id)
        if item and item.url:
            st.markdown(
                f'<span class="source-chip">{source_id}</span> '
                f"[{item.title or item.url}]({item.url})",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="source-chip">{source_id}</span>',
                unsafe_allow_html=True,
            )
        if item:
            quoted = item.snippet[:220] + "..." if len(item.snippet) > 220 else item.snippet
            st.caption(f'"{quoted}"')


def render_evidence(scout: ScoutReport) -> None:
    """List Agent 1's queries and its retrieved snippets."""
    st.markdown("**Search queries generated**")
    for index, query in enumerate(scout.queries, start=1):
        st.markdown(f"{index}. `{query}`")

    if not scout.evidence:
        st.warning("No evidence snippets were retrieved.")
        return

    st.markdown(f"**Evidence retrieved &mdash; {len(scout.evidence)} snippet(s)**")
    for item in scout.evidence:
        header = f"{item.source_id} - {item.title or 'Untitled result'}"
        with st.expander(header, expanded=False):
            st.write(item.snippet)
            if item.url:
                st.markdown(f"[Open source]({item.url})")
            if item.query:
                st.caption(f"Found via: {item.query}")


def render_alignments(audit: AuditReport, scout: ScoutReport) -> None:
    """Render Agent 2's badge-coded NLI labels."""
    if not audit.alignments:
        st.warning("No snippets to audit.")
        return

    tally = audit.tally()
    cols = st.columns(3)
    cols[0].metric("🟢 Supports", tally["SUPPORTS"])
    cols[1].metric("🔴 Refutes", tally["REFUTES"])
    cols[2].metric("🟡 Not enough", tally["NOT_ENOUGH_INFO"])

    lookup = scout.by_id()
    for alignment in audit.alignments:
        style = style_for(alignment.verdict)
        item = lookup.get(alignment.source_id)
        title = item.title if item and item.title else ""
        title_html = f'<br/><span class="muted">{title}</span>' if title else ""
        st.markdown(
            f"""
            <div class="nli-row" style="border-left-color:{style['color']}">
                <span class="nli-id">{alignment.source_id}</span>
                &nbsp;{style['emoji']}
                <strong style="color:{style['color']}">{style['label']}</strong><br/>
                {alignment.justification}{title_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_static_trace(result: MultiAgentResult) -> None:
    """Re-render a completed run's trace on a rerun (no live status boxes)."""
    with st.expander("1. Agent 1 - Evidence Scout", expanded=False):
        render_evidence(result.scout)
    with st.expander("2. Agent 2 - Alignment Auditor", expanded=False):
        render_alignments(result.audit, result.scout)
    with st.expander("3. Agent 3 - Synthesis Adjudicator", expanded=False):
        st.write(result.adjudication.detailed_reasoning)


def render_timings(result: MultiAgentResult) -> None:
    """Show per-stage latency and any retrieval warnings."""
    with st.expander("Stage timings & diagnostics", expanded=False):
        for timing in result.timings:
            st.markdown(f"- {timing.name}: `{timing.seconds:.2f}s`")
        st.markdown(f"- **Total: `{result.total_seconds:.2f}s`**")
        for warning in result.warnings:
            st.warning(warning)


def render_comparison(multi: MultiAgentResult, baseline: BaselineResult) -> None:
    """Highlight where the grounded and ungrounded verdicts diverge."""
    grounded = multi.adjudication
    ungrounded = baseline.adjudication
    delta = grounded.confidence - ungrounded.confidence

    st.markdown("### Head-to-head")
    cols = st.columns(3)
    cols[0].metric("Multi-agent", style_for(grounded.final_verdict)["label"])
    cols[1].metric("Baseline", style_for(ungrounded.final_verdict)["label"])
    cols[2].metric(
        "Multi-agent confidence",
        f"{grounded.confidence:.0%}",
        delta=f"{delta:+.0%} vs baseline",
    )

    if grounded.final_verdict != ungrounded.final_verdict:
        st.error(
            f"**Verdicts disagree.** The grounded pipeline returned "
            f"`{grounded.final_verdict}` citing {len(grounded.cited_sources)} "
            f"source(s); the ungrounded baseline returned "
            f"`{ungrounded.final_verdict}` from memory with no sources. "
            "Divergence here is the signal this benchmark exists to expose."
        )
    elif ungrounded.confidence > grounded.confidence + 0.15:
        st.warning(
            "**Same verdict, overconfident baseline.** The baseline is more "
            f"certain ({ungrounded.confidence:.0%}) than the evidence-grounded "
            f"pipeline ({grounded.confidence:.0%}) despite citing nothing."
        )
    else:
        st.success("Both paths agree, and the baseline is not overconfident.")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def render_sidebar() -> Settings:
    """Draw the sidebar and return settings resolved from the environment.

    Configuration is environment-driven (see ``.env.example``) rather than a
    panel of widgets, so the sidebar explains the pipeline, loads examples,
    and reports backend health.
    """
    st.sidebar.title("🔎 Fact-Checker")
    st.sidebar.markdown(
        "Three specialised agents verify a claim against **live web evidence**, "
        "benchmarked against a single LLM answering **from memory alone**."
    )

    st.sidebar.markdown(
        """
**How it works**

1. **Evidence Scout** - writes 2 neutral queries, searches the web, returns
   snippets as `doc_1 ... doc_n`.
2. **Alignment Auditor** - labels each snippet `SUPPORTS` / `REFUTES` /
   `NOT_ENOUGH_INFO` using *only* that snippet's text.
3. **Synthesis Adjudicator** - weighs the labels into one verdict with a
   calibrated confidence and explicit citations.
4. **Baseline** - one zero-shot call, no search, same output schema.
"""
    )

    settings = Settings()

    st.sidebar.divider()
    st.sidebar.subheader("Verdict aggregation")
    deterministic = st.sidebar.toggle(
        "Compute the verdict from the audit",
        value=settings.deterministic_aggregation,
        help=(
            "On: the label and confidence are computed from the per-snippet "
            "audit by the weighting policy, in code, and the agent's own "
            "label is overridden. Off: Agent 3 decides, as described in the "
            "paper's reference condition. The agent writes the reasoning "
            "either way."
        ),
    )
    settings.deterministic_aggregation = deterministic
    if deterministic:
        settings.aggregation_beta = st.sidebar.slider(
            "Specificity weight β",
            min_value=0.0,
            max_value=3.0,
            value=float(settings.aggregation_beta),
            step=0.5,
            help=(
                "How much more a snippet counts when its justification quotes "
                "a figure, date, or official record. 0 is unweighted voting."
            ),
        )
        settings.gate_citations = st.sidebar.checkbox(
            "Drop citations the audit contradicts",
            value=settings.gate_citations,
            help=(
                "Keep a citation only when the audit labelled that snippet in "
                "the same direction as the verdict."
            ),
        )

    st.sidebar.divider()
    st.sidebar.subheader("Quick-load examples")
    for button_label, help_text, text in EXAMPLE_CLAIMS:
        st.sidebar.button(
            button_label,
            help=help_text,
            use_container_width=True,
            on_click=load_example,
            args=(text,),
        )

    return settings


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def execute_pipeline(claim: str, settings: Settings) -> Optional[MultiAgentResult]:
    """Run the three agents live, streaming progress into st.status blocks."""
    pipeline = FactCheckPipeline(settings)
    pipeline.reset()

    try:
        with st.status("1. Agent 1 - Evidence Scout", expanded=True) as status:
            st.caption("Generating neutral queries and searching the web...")
            scout = pipeline.scout(claim)
            render_evidence(scout)
            status.update(
                label=f"1. Agent 1 - Evidence Scout ({len(scout.evidence)} snippets)",
                state="complete",
                expanded=False,
            )

        for warning in pipeline.warnings:
            st.warning(warning)

        with st.status("2. Agent 2 - Alignment Auditor", expanded=True) as status:
            st.caption("Labelling each snippet independently against the claim...")
            audit = pipeline.audit(claim, scout)
            render_alignments(audit, scout)
            status.update(
                label=(
                    f"2. Agent 2 - Alignment Auditor "
                    f"({len(audit.alignments)} judgements)"
                ),
                state="complete",
                expanded=False,
            )

        with st.status("3. Agent 3 - Synthesis Adjudicator", expanded=True) as status:
            st.caption("Weighing the audit into a single calibrated verdict...")
            adjudication = pipeline.adjudicate(claim, scout, audit)
            st.write(adjudication.detailed_reasoning)
            if pipeline.trace is not None:
                # Showing the arithmetic is the point of computing the verdict
                # rather than asking for it: the decision is reconstructible.
                st.caption(f"Computed aggregation — {pipeline.trace.explain()}")
                if pipeline.trace.replaced_verdict != adjudication.final_verdict:
                    st.caption(
                        f"The agent proposed {pipeline.trace.replaced_verdict} at "
                        f"confidence {pipeline.trace.replaced_confidence:.2f}; the "
                        "computed policy overrode it."
                    )
            status.update(
                label=(
                    f"3. Agent 3 - Synthesis Adjudicator "
                    f"({adjudication.final_verdict})"
                ),
                state="complete",
                expanded=False,
            )
    except (AgentError, ConfigurationError) as exc:
        st.session_state.run_error = str(exc)
        return None

    return MultiAgentResult(
        claim=claim,
        scout=scout,
        audit=audit,
        adjudication=adjudication,
        timings=list(pipeline.timings),
        warnings=list(pipeline.warnings),
    )


def execute_baseline(claim: str, settings: Settings) -> Optional[BaselineResult]:
    """Run the ungrounded control call."""
    try:
        with st.spinner("Answering from parametric memory only..."):
            return FactCheckPipeline(settings).baseline(claim)
    except (AgentError, ConfigurationError) as exc:
        st.session_state.baseline_error = str(exc)
        return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    """Compose the page."""
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    settings = render_sidebar()

    st.title("🔎 Multi-Agent Fact-Checking Assistant")
    st.caption(
        "Retrieval-grounded multi-agent verification versus a monolithic "
        "single-LLM control - same claim, same output schema, side by side."
    )

    st.text_area(
        "Claim to verify",
        key="claim_text",
        height=120,
        placeholder=(
            "e.g. The Great Wall of China is visible from the Moon "
            "with the naked eye."
        ),
    )

    run_clicked = st.button("Verify Claim", type="primary", use_container_width=True)

    claim = str(st.session_state.claim_text or "").strip()

    st.divider()
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Multi-Agent System")
        st.caption("Grounded in live web evidence")
        left_trace = st.container()
        left_verdict = st.container()

    with col_right:
        st.subheader("Single-LLM Baseline")
        st.caption("Zero-shot, no search, internal knowledge only")
        right_body = st.container()

    comparison_slot = st.container()

    if run_clicked:
        if not claim:
            st.warning("Enter a claim before running the check.")
            return
        if not settings.is_configured:
            st.error(
                "No Anthropic API key configured. Add one in the sidebar, set "
                "ANTHROPIC_API_KEY in your environment, or switch the backend "
                "to Ollama to run locally."
            )
            return

        # Pre-flight the local server so a dead Ollama fails here with a clear
        # fix, rather than part-way through Agent 1.
        if settings.is_local:
            reachable, installed = ollama_status(settings.ollama_host)
            if not reachable:
                st.error(
                    f"**No Ollama server at {settings.ollama_host}.**\n\n"
                    "1. Install it from https://ollama.com\n"
                    f"2. Pull the model: `ollama pull {settings.model}`\n"
                    "3. Verify with `ollama list`\n\n"
                    "Or switch **Model backend** to the Anthropic API in the sidebar."
                )
                return
            if installed and settings.model not in installed:
                st.error(
                    f"**Model `{settings.model}` is not pulled.**\n\n"
                    f"Run: `ollama pull {settings.model}`\n\n"
                    f"Installed: {', '.join(installed) or '(none)'}"
                )
                return

        # Clear previous output so a failed run cannot show stale results.
        st.session_state.run_error = None
        st.session_state.baseline_error = None
        st.session_state.multi_result = None
        st.session_state.baseline_result = None

        with left_trace:
            st.session_state.multi_result = execute_pipeline(claim, settings)
        with right_body:
            st.session_state.baseline_result = execute_baseline(claim, settings)
        st.session_state.just_ran = True
    else:
        st.session_state.just_ran = False

    multi: Optional[MultiAgentResult] = st.session_state.multi_result
    baseline: Optional[BaselineResult] = st.session_state.baseline_result

    # ---- left column output ----
    with left_trace:
        if st.session_state.run_error:
            st.error(f"Pipeline failed: {st.session_state.run_error}")
        elif multi is None:
            st.info("Submit a claim to run the three-agent pipeline.")
        elif not st.session_state.just_ran:
            render_static_trace(multi)

    with left_verdict:
        if multi is not None:
            render_verdict_card(
                multi.adjudication,
                title="Multi-agent verdict - evidence-grounded",
                scout=multi.scout,
            )
            render_timings(multi)

    # ---- right column output ----
    with right_body:
        if st.session_state.baseline_error:
            st.error(f"Baseline failed: {st.session_state.baseline_error}")
        elif baseline is None:
            st.info("The baseline verdict appears here for comparison.")
        else:
            render_verdict_card(
                baseline.adjudication,
                title="Baseline verdict - no retrieval",
                scout=None,
            )
            st.caption(
                f"Completed in {baseline.seconds:.2f}s - 1 LLM call - 0 sources"
            )

    # ---- comparison ----
    with comparison_slot:
        if multi is not None and baseline is not None:
            st.divider()
            render_comparison(multi, baseline)


if __name__ == "__main__":
    main()
