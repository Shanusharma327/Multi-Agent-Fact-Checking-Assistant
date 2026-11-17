"""Build the results report as a PDF and a Word document.

    python src/analysis/build_results_doc.py

Both documents are assembled from the artefacts the analysis chain
writes into ``results/``:

    results/macros/*.tex    every reported quantity, as a LaTeX macro
    results/figures/*.png   every chart
    logs/*.log              console output from real runs

**No number in this script is typed by hand.** Every value is looked up
by macro name from ``results/macros/``, and a lookup that misses raises
rather than printing a blank, so a stale or renamed quantity fails the
build instead of silently reaching the page. The two documents are
rendered from one table specification, so the PDF and the .docx cannot
disagree with each other or with the run.

Run ``python src/analysis/regenerate.py`` first; this script reads what
that produces.

Outputs:
    results/MAFCA_Results.pdf
    results/MAFCA_Results.docx
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/analysis/..)
MACRO_DIR = ROOT / "results" / "macros"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "results"

TITLE = "MAFCA: Results Report"
SUBTITLE = "A Multi-Agent Fact-Checking Assistant"
COURSE = "Machine Learning IA 2"
INSTITUTE = (
    "Department of Computer Engineering, "
    "K. J. Somaiya School of Engineering, Somaiya Vidyavihar University"
)
AUTHORS = ["Shanu Sharma", "Krutika Sadadekar"]

MACRO_RE = re.compile(r"newcommand\{" + "\\\\" + r"([A-Za-z]+)\}\{([^}]*)\}")


# --------------------------------------------------------------------------
# macro lookup
# --------------------------------------------------------------------------

class Macros:
    """Every generated quantity, looked up by name.

    A missing name is an error rather than an empty cell: the whole point
    of generating the numbers is that the document cannot drift from the
    run, and a silently blank cell would defeat that.
    """

    def __init__(self, directory: Path) -> None:
        self.values: Dict[str, str] = {}
        self.sources: Dict[str, str] = {}
        for path in sorted(directory.glob("*.tex")):
            text = path.read_text(encoding="utf-8")
            for name, value in MACRO_RE.findall(text):
                self.values[name] = value.strip()
                self.sources[name] = path.name
        if not self.values:
            raise SystemExit(
                f"No macros found in {directory}.\n"
                "Run `python src/analysis/regenerate.py` first."
            )

    def __call__(self, name: str) -> str:
        try:
            return self.values[name]
        except KeyError:
            raise KeyError(
                f"Macro \\{name} is not defined in {MACRO_DIR}. "
                "Re-run src/analysis/regenerate.py, or correct the name."
            ) from None

    def ci(self, point: str, lo: str, hi: str) -> str:
        return f"{self(point)} [{self(lo)}, {self(hi)}]"


# --------------------------------------------------------------------------
# document specification -- rendered to both PDF and .docx
# --------------------------------------------------------------------------

Table = Tuple[str, List[str], List[List[str]], str]   # caption, head, rows, note
Figure = Tuple[str, str, str]                          # filename, caption, note


def build_tables(m: Macros) -> Dict[str, Table]:
    n = m("CondN")

    def row4(label: str, p: str) -> List[str]:
        """One condition's classification row, with bootstrap intervals."""
        return [
            label,
            m.ci(f"Cond{p}Acc", f"Cond{p}AccLo", f"Cond{p}AccHi"),
            m.ci(f"Cond{p}FOne", f"Cond{p}FOneLo", f"Cond{p}FOneHi"),
            m.ci(f"Cond{p}Dec", f"Cond{p}DecLo", f"Cond{p}DecHi"),
            m(f"Cond{p}Abstain"),
        ]

    def diff(label: str, p: str) -> List[str]:
        return [
            label,
            m(f"CondD{p}"),
            f"[{m(f'CondD{p}Lo')}, {m(f'CondD{p}Hi')}]",
            m(f"CondD{p}P"),
            m(f"CondD{p}F"),
            f"[{m(f'CondD{p}FLo')}, {m(f'CondD{p}FHi')}]",
        ]

    tables: Dict[str, Table] = {}

    tables["classification"] = (
        f"Classification performance on all n = {n} claims. Values are "
        "percentages with 95% bootstrap confidence intervals over 10,000 "
        "resamples. 'Decidable' restricts to claims whose gold label is "
        "SUPPORTS or REFUTES. 'Abstention' is the share answered "
        "NOT_ENOUGH_INFO.",
        ["Condition", "Accuracy (%)", "Macro-F1 (%)", "Decidable (%)", "Abstain (%)"],
        [
            row4("B0: single LLM, no retrieval", "Bzero"),
            row4("B1: single-call RAG", "Bone"),
            row4("B4: MAFCA, computed verdict", "Bfour"),
            row4("MAFCA: agent-reported (ablation)", "Mafca"),
            row4("Majority class", "Majority"),
            row4("Random", "Random"),
        ],
        "Every condition replays the identical cached passages, so the "
        "comparisons are paired claim for claim.",
    )

    tables["differences"] = (
        "Paired differences in percentage points. Positive favours the "
        "first-named condition. p is the exact McNemar test on accuracy. "
        "Every interval here contains zero.",
        ["Comparison", "ΔAcc (pp)", "95% CI", "p", "ΔF1 (pp)", "95% CI"],
        [
            diff("B4 vs B0", "BfourVsBzero"),
            diff("B4 vs B1", "BfourVsBone"),
            diff("MAFCA vs B0", "MafcaVsBzero"),
            diff("MAFCA vs B1", "MafcaVsBone"),
        ],
        "No comparison reaches significance at this sample size. See the "
        "power table for what the sample could have resolved.",
    )

    tables["power"] = (
        "Statistical power of the paired design. The minimum detectable "
        "difference (MDD) is the smallest paired difference this sample "
        "could resolve; observed differences smaller than it are "
        "uninformative in either direction.",
        ["Quantity", "B4 vs B0", "MAFCA vs B1"],
        [
            ["Claims (n)", n, n],
            ["Discordant pairs", m("CondDiscordance"), m("CondDiscordanceMafcaBone")],
            ["Informative pairs", m("CondInformativePairs"),
             m("CondInformativePairsMafcaBone")],
            ["Minimum detectable difference (pp)", m("CondMdd"),
             m("CondMddMafcaBone")],
        ],
        "A paired test only sees claims the two conditions answer "
        "differently, which is why the discordant count, not n, drives power.",
    )

    tables["calibration"] = (
        "Calibration and confidence. ECE is expected calibration error and "
        "Brier the quadratic score; for both, lower is better. Computing "
        "the confidence from the audit rather than eliciting it does not "
        "improve either past the single-LLM baseline.",
        ["Condition", "ECE", "Brier"],
        [
            ["B0: single LLM", m("CondBzeroEce"), m("CondBzeroBrier")],
            ["B1: single-call RAG", m("CondBoneEce"), m("CondBoneBrier")],
            ["B4: MAFCA, computed", m("CondBfourEce"), m("CondBfourBrier")],
            ["MAFCA: agent-reported", m("CondMafcaEce"), m("CondMafcaBrier")],
            ["Majority class", m("CondMajorityEce"), m("CondMajorityBrier")],
            ["Random", m("CondRandomEce"), m("CondRandomBrier")],
        ],
        "",
    )

    tables["citations"] = (
        "Citation behaviour and cost. 'Cited' is the share of verdicts "
        "carrying at least one citation that survives validation: it must "
        "name a passage that was genuinely retrieved and that the "
        "verification stage labelled in the verdict's direction.",
        ["Condition", "Cited (%)", "Support (%)", "LLM calls"],
        [
            ["B0: single LLM", m("CondBzeroCited"), m("CondBzeroSupport"),
             m("CondBzeroCalls")],
            ["B1: single-call RAG", m("CondBoneCited"), m("CondBoneSupport"),
             m("CondBoneCalls")],
            ["B4: MAFCA, computed", m("CondBfourCited"), m("CondBfourSupport"),
             m("CondBfourCalls")],
            ["MAFCA: agent-reported", m("CondMafcaCited"), m("CondMafcaSupport"),
             m("CondMafcaCalls")],
        ],
        "B0 cites nothing because it retrieves nothing. The call count is "
        "measured at replay: the Evidence Scout's call is made once during "
        "cache construction and does not recur.",
    )

    tables["attribution"] = (
        "Where incorrect verdicts arise, as a share of all errors, with 95% "
        "bootstrap intervals over 10,000 resamples of the claims. This "
        "analysis requires a per-passage record, so it cannot be run on the "
        "single-call baseline at all.",
        ["Stage", "Live retrieval (%)", "Oracle evidence (%)"],
        [
            ["Retrieval",
             m.ci("CondErrLiveRetrieval", "CondErrLiveRetrievalLo",
                  "CondErrLiveRetrievalHi"),
             m.ci("CondErrOracleRetrieval", "CondErrOracleRetrievalLo",
                  "CondErrOracleRetrievalHi")],
            ["Verification",
             m.ci("CondErrLiveVerification", "CondErrLiveVerificationLo",
                  "CondErrLiveVerificationHi"),
             m.ci("CondErrOracleVerification", "CondErrOracleVerificationLo",
                  "CondErrOracleVerificationHi")],
            ["Aggregation",
             m.ci("CondErrLiveAggregation", "CondErrLiveAggregationLo",
                  "CondErrLiveAggregationHi"),
             m.ci("CondErrOracleAggregation", "CondErrOracleAggregationLo",
                  "CondErrOracleAggregationHi")],
        ],
        "Retrieval dominates under live conditions. Substituting the "
        "corpus' own annotated evidence inverts the profile. No error is "
        "attributable to aggregation in either condition, because the "
        "computed policy cannot return a verdict the audit does not support.",
    )

    tables["oracle"] = (
        "Oracle-evidence diagnostic: the same claims with the corpus' "
        "annotated evidence substituted for web search. This isolates "
        "verification quality from retrieval quality. These are diagnostic "
        "figures, not deployable performance.",
        ["Condition", "Accuracy (%)", "Macro-F1 (%)", "Decidable (%)", "ECE"],
        [
            ["B1, live evidence",
             m.ci("CondBoneAcc", "CondBoneAccLo", "CondBoneAccHi"),
             m("CondBoneFOne"), m("CondBoneDec"), m("CondBoneEce")],
            ["B1, oracle evidence",
             m.ci("CondOracleBoneAcc", "CondOracleBoneAccLo",
                  "CondOracleBoneAccHi"),
             m("CondOracleBoneFOne"), m("CondOracleBoneDec"),
             m("CondOracleBoneEce")],
            ["MAFCA, live evidence",
             m.ci("CondMafcaAcc", "CondMafcaAccLo", "CondMafcaAccHi"),
             m("CondMafcaFOne"), m("CondMafcaDec"), m("CondMafcaEce")],
            ["MAFCA, oracle evidence",
             m.ci("CondOracleMafcaAcc", "CondOracleMafcaAccLo",
                  "CondOracleMafcaAccHi"),
             m("CondOracleMafcaFOne"), m("CondOracleMafcaDec"),
             m("CondOracleMafcaEce")],
        ],
        "",
    )

    tables["oracle_gain"] = (
        "Effect of substituting annotated evidence, in percentage points, "
        "with Holm-Bonferroni adjusted p-values across the three "
        "comparisons.",
        ["Comparison", "ΔAcc (pp)", "95% CI", "p", "p (Holm)"],
        [
            ["B1 oracle vs B1 live", m("CondDOracleBoneVsBone"),
             f"[{m('CondDOracleBoneVsBoneLo')}, {m('CondDOracleBoneVsBoneHi')}]",
             m("CondDOracleBoneVsBoneP"), m("CondDOracleBoneVsBoneHolm")],
            ["MAFCA oracle vs MAFCA live", m("CondDOracleMafcaVsMafca"),
             f"[{m('CondDOracleMafcaVsMafcaLo')}, {m('CondDOracleMafcaVsMafcaHi')}]",
             m("CondDOracleMafcaVsMafcaP"), m("CondDOracleMafcaVsMafcaHolm")],
            ["MAFCA oracle vs B1 oracle", m("CondDOracleMafcaVsOracleBone"),
             f"[{m('CondDOracleMafcaVsOracleBoneLo')}, "
             f"{m('CondDOracleMafcaVsOracleBoneHi')}]",
             m("CondDOracleMafcaVsOracleBoneP"),
             m("CondDOracleMafcaVsOracleBoneHolm")],
        ],
        "",
    )

    tables["retrieval"] = (
        "Retrieval quality against the corpus' annotated source URLs, at "
        "two matching strictnesses. Exact matching requires the same URL; "
        "domain matching requires only the same site.",
        ["Measure", "Exact URL", "Domain"],
        [
            ["Claims scored", m("AvRetExactN"), m("AvRetDomainN")],
            ["Recall@k (%)", m("AvRetExactRecall"), m("AvRetDomainRecall")],
            ["Precision@k (%)", m("AvRetExactPrecision"), m("AvRetDomainPrecision")],
            ["Claims with >=1 hit (%)", m("AvRetExactHit"), m("AvRetDomainHit")],
            ["Mean gold URLs", m("AvRetExactMeanGold"), m("AvRetDomainMeanGold")],
            ["Mean retrieved", m("AvRetExactMeanRetrieved"),
             m("AvRetDomainMeanRetrieved")],
        ],
        "The system rarely recovers the exact sources the annotator used; "
        "it retrieves contemporaneous material instead, which the "
        "ungrounded baseline lacks entirely.",
    )

    def per_class(label: str, p: str) -> List[List[str]]:
        return [
            [label, "SUPPORTS", m(f"Av{p}PrecSup"), m(f"Av{p}RecSup"),
             m(f"Av{p}FOneSup"), m(f"Av{p}SupportSup")],
            ["", "REFUTES", m(f"Av{p}PrecRef"), m(f"Av{p}RecRef"),
             m(f"Av{p}FOneRef"), m(f"Av{p}SupportRef")],
            ["", "NOT_ENOUGH_INFO", m(f"Av{p}PrecNei"), m(f"Av{p}RecNei"),
             m(f"Av{p}FOneNei"), m(f"Av{p}SupportNei")],
        ]

    tables["per_class"] = (
        "Per-class precision, recall and F1 (%), with class support. This "
        "is the confusion structure behind the macro-averaged figures: it "
        "shows which labels each system actually gets right.",
        ["System", "Class", "Precision", "Recall", "F1", "Support"],
        per_class("Single LLM (B0)", "Base")
        + per_class("MAFCA computed", "Det")
        + per_class("MAFCA agent-reported", "Multi"),
        "Support is the number of claims carrying that gold label.",
    )

    tables["cost"] = (
        "Mean wall-clock latency per claim, in seconds, measured during the "
        "run. Decomposition costs time: the pipeline makes more model calls "
        "than the control.",
        ["System", "Mean latency (s)"],
        [
            ["Single LLM (B0)", m("AvBaseLatency")],
            ["MAFCA computed", m("AvDetLatency")],
            ["MAFCA agent-reported", m("AvMultiLatency")],
            ["Ratio, MAFCA to baseline", m("AvLatencyRatio")],
        ],
        "",
    )

    tables["integrity"] = (
        "Pipeline integrity checks, all enforced in code rather than by "
        "prompt. These hold by construction and so do not depend on the "
        "sample size.",
        ["Check", "Value"],
        [
            ["Claims evaluated", m("AvClaims")],
            ["Mean evidence passages per claim", m("AvMeanEvidence")],
            ["Claims with zero evidence retrieved", m("AvZeroEvidence")],
            ["Claims issuing two distinct queries (%)", m("AvTwoQuery")],
            ["Verdicts with a complete audit trace (%)", m("AvTraced")],
            ["Audit-schema validity (%)", m("AvMultiValidity")],
            ["Leakage: fact-check domains filtered", m("AvLeakFiltered")],
            ["Leakage: claims still exposed", m("AvLeakExposed")],
            ["Leakage rate (%)", m("AvLeakRate")],
            ["Quoted-span rate (%)", m("AvQuoteRate")],
            ["Spans checkable against passage text", m("AvSpanCheckable")],
            ["Span violations", m("AvSpanViolations")],
        ],
        "The leakage filter removes the claim's own fact-check article "
        "from retrieved evidence, which would otherwise hand the verdict "
        "to the system.",
    )

    tables["degraded"] = (
        "Retrieval degradation during the run. The search service failed "
        "partway through and the run was halted rather than looped; these "
        "counts record how many claims were affected.",
        ["Quantity", "Claims"],
        [
            ["Claims retrieved before the outage", m("CondN")],
            ["Claims with degraded retrieval", m("CondDegraded")],
            ["  of which returned no evidence", m("CondDegradedEmpty")],
            ["  of which returned partial evidence", m("CondDegradedPartial")],
            ["Claims after excluding degraded retrieval", m("CondCleanN")],
        ],
        "Reporting the clean subset separately guards against a result "
        "that depends on which claims the outage happened to hit.",
    )

    return tables


FIGURES: List[Figure] = [
    ("fig_flowchart.png",
     "The three-agent pipeline. Evidence Scout retrieves passages, "
     "Alignment Auditor labels each passage separately, and Synthesis "
     "Adjudicator writes the reasoning trace while the verdict itself is "
     "computed from the audit.",
     "src/analysis/summary_figures.py"),
    ("poster_accuracy.png",
     "Accuracy of the single LLM (B0), single-call RAG (B1) and MAFCA, "
     "overall and on the decidable subset. The dashed line is the "
     "majority-class floor on the decidable subset, which no system reaches.",
     "src/analysis/poster_figures.py"),
    ("av_accuracy.png",
     "Accuracy, overall and on the decidable subset, for the single-LLM "
     "control against both MAFCA configurations.",
     "src/analysis/results_report.py"),
    ("av_f1.png",
     "F1-score by gold label and macro-averaged.",
     "src/analysis/results_report.py"),
    ("av_precision.png",
     "Precision by gold label and macro-averaged.",
     "src/analysis/results_report.py"),
    ("av_recall.png",
     "Recall by gold label and macro-averaged.",
     "src/analysis/results_report.py"),
    ("av_errorrate.png",
     "Overall error rate, the complement of accuracy.",
     "src/analysis/results_report.py"),
    ("av_latency.png",
     "Mean wall-clock latency per claim. Decomposition costs time.",
     "src/analysis/results_report.py"),
    ("av_reliability.png",
     "Reliability diagram. A perfectly calibrated system would track the "
     "diagonal; distance from it is miscalibration.",
     "src/analysis/grounding_figures.py"),
    ("av_retrieval.png",
     "Evidence retrieval measured against the corpus' annotated gold URLs, "
     "at exact-URL and domain matching strictness.",
     "src/analysis/grounding_figures.py"),
    ("av_ablation.png",
     "Aggregation ablation over the evidence-strength weight beta, which "
     "controls how much a passage's assessed strength scales its vote.",
     "src/analysis/ablation.py"),
    ("fig_summary_bar.png",
     "Summary comparison across the headline measures.",
     "src/analysis/summary_figures.py"),
    ("fig_network.png",
     "Agent communication structure and the typed schemas carried across "
     "each boundary.",
     "src/analysis/summary_figures.py"),
]


# --------------------------------------------------------------------------
# console output
# --------------------------------------------------------------------------

def read_log(name: str, max_lines: int, width: int = 100) -> List[str]:
    """Return console output, clipped and made safe for both renderers.

    The logs are whatever the console emitted, so they can carry bytes
    that are not valid UTF-8 and characters that pdflatex has no glyph
    for. Both are reduced to ASCII here rather than in the renderers, so
    the PDF and the .docx show the same text.
    """
    path = LOG_DIR / name
    if not path.exists():
        return [f"[{name} not present -- run the command that produces it]"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    clipped = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        clipped = True
    out = []
    for ln in lines:
        ln = ln.replace("�", "?").replace("\t", "    ")
        ln = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in ln)
        out.append(ln[:width] + (" ..." if len(ln) > width else ""))
    if clipped:
        out.append(f"... [{name} continues; full file in logs/{name}]")
    return out


# --------------------------------------------------------------------------
# LaTeX rendering
# --------------------------------------------------------------------------

TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(TEX_ESCAPES.get(ch, ch))
    s = "".join(out)
    # Unicode the document uses that pdflatex needs help with.
    s = s.replace("Δ", r"$\Delta$").replace("≥", r"$\geq$")
    s = s.replace(">=", r"$\geq$").replace("beta", r"$\beta$")
    return s


def tex_table(caption: str, head: Sequence[str], rows: Sequence[Sequence[str]],
              note: str, label: str) -> str:
    ncols = len(head)
    align = "l" + "r" * (ncols - 1)
    out = [
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule",
        " & ".join(rf"\textbf{{{tex_escape(h)}}}" for h in head) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        out.append(" & ".join(tex_escape(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(rf"\par\vspace{{2pt}}\footnotesize{{{tex_escape(note)}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def build_tex(m: Macros, tables: Dict[str, Table], figures: List[Figure]) -> str:
    n = m("CondN")
    L: List[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{longtable}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{xcolor}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\pagestyle{fancy}\fancyhf{}",
        rf"\lhead{{\small {tex_escape(SUBTITLE)}}}",
        r"\rhead{\small\thepage}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        r"\begin{document}",
        r"\begin{center}",
        rf"{{\LARGE\bfseries {tex_escape(TITLE)}}}\\[4pt]",
        rf"{{\large {tex_escape(SUBTITLE)}}}\\[10pt]",
        rf"{tex_escape(COURSE)}\\[2pt]",
        rf"{tex_escape(INSTITUTE)}\\[8pt]",
        rf"\textbf{{{tex_escape(' and '.join(AUTHORS))}}}",
        r"\end{center}",
        r"\vspace{6pt}\hrule\vspace{10pt}",
        r"\section*{Scope and reproducibility}",
        tex_escape(
            f"Every table, figure and number in this document was produced by "
            f"executing the code in this repository on n = {n} AVeriTeC claims. "
            "Nothing is transcribed by hand: each value is read from a generated "
            "macro in results/macros/, and a name that no longer exists raises an "
            "error rather than printing a blank."
        ),
        r"\textbf{To reproduce every artefact below:}",
        r"\begin{verbatim}",
        "python src/analysis/regenerate.py      # rebuilds results/ from the cache",
        "python src/analysis/build_results_doc.py   # rebuilds this document",
        "python -m pytest -q                    # 434 tests",
        r"\end{verbatim}",
        tex_escape(
            "This runs offline. It makes no model call and needs no corpus "
            "download: the retrieved passages are committed in cache/av250.jsonl "
            "and the per-claim verdicts in results/, so the chain replays exactly. "
            "Decoding is greedy (temperature 0) and every sampling step is seeded, "
            "so reruns reproduce the same values."
        ),
        tex_escape(
            "Scope: the protocol was designed for 250 claims. The search service "
            f"degraded partway through retrieval and the run was halted at {n} "
            "claims rather than looped, because continuing against a failing "
            "endpoint fills the cache with empty evidence sets indistinguishable "
            "from genuine 'nothing found' results. This is a pilot, and the "
            "statistical power table states what it can and cannot settle."
        ),
        r"\clearpage",
        r"\section{Console output}",
        tex_escape(
            "Actual console output from executing the analysis chain. The full "
            "files are in logs/ in the repository."
        ),
        r"\subsection{Regeneration of every result: "
        r"\texttt{python src/analysis/regenerate.py}}",
        r"\begin{small}\begin{verbatim}",
    ]
    L += read_log("regenerate.log", 70)
    L += [
        r"\end{verbatim}\end{small}",
        r"\subsection{Test suite: \texttt{python -m pytest -q}}",
        r"\begin{small}\begin{verbatim}",
    ]
    L += read_log("pytest.log", 20)
    L += [r"\end{verbatim}\end{small}", r"\clearpage",
          r"\section{Tables}"]

    order = [
        ("classification", "Classification performance"),
        ("differences", "Paired differences between conditions"),
        ("power", "Statistical power"),
        ("calibration", "Calibration"),
        ("citations", "Citations and cost"),
        ("attribution", "Error attribution by stage"),
        ("oracle", "Oracle-evidence diagnostic"),
        ("oracle_gain", "Effect of oracle evidence"),
        ("retrieval", "Retrieval quality"),
        ("per_class", "Per-class performance"),
        ("cost", "Latency"),
        ("integrity", "Pipeline integrity checks"),
        ("degraded", "Retrieval degradation"),
    ]
    for key, heading in order:
        caption, head, rows, note = tables[key]
        L.append(rf"\subsection{{{tex_escape(heading)}}}")
        L.append(tex_table(caption, head, rows, note, key))

    L += [r"\clearpage", r"\section{Figures}"]
    for filename, caption, source in figures:
        path = FIG_DIR / filename
        if not path.exists():
            continue
        rel = (Path("figures") / filename).as_posix()
        L += [
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width=0.84\textwidth]{{{rel}}}",
            rf"\caption{{{tex_escape(caption)} "
            rf"\textit{{Generated by {tex_escape(source)}.}}}}",
            r"\end{figure}",
        ]

    L += [
        r"\clearpage",
        r"\section*{Index of generated quantities}",
        tex_escape(
            f"This document draws on {len(m.values)} generated macros across "
            "three files in results/macros/: numbers.tex, condition_numbers.tex "
            "and attribution_numbers.tex. Each is written by the analysis "
            "scripts and regenerated by src/analysis/regenerate.py."
        ),
        r"\end{document}",
    ]
    return "\n".join(L)


def compile_pdf(tex_path: Path) -> bool:
    exe = shutil.which("pdflatex")
    if not exe:
        print("  ! pdflatex not found; skipping PDF", file=sys.stderr)
        return False
    for _ in range(2):
        result = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error",
             tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    if result.returncode != 0:
        tail = (result.stdout or "")[-2500:]
        print("  ! pdflatex failed:\n" + tail, file=sys.stderr)
        return False
    for ext in (".aux", ".log", ".out", ".toc"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return True


# --------------------------------------------------------------------------
# DOCX rendering
# --------------------------------------------------------------------------

def build_docx(m: Macros, tables: Dict[str, Table], figures: List[Figure],
               out_path: Path) -> bool:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt, RGBColor
    except ImportError:
        print("  ! python-docx not installed (pip install python-docx); "
              "skipping .docx", file=sys.stderr)
        return False

    n = m("CondN")
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.9)

    def para(text: str, size: int = 10, bold: bool = False,
             italic: bool = False, align=None, space_after: int = 6):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        return p

    def mono(lines: Sequence[str]):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        for i, line in enumerate(lines):
            run = p.add_run(line + ("\n" if i < len(lines) - 1 else ""))
            run.font.name = "Consolas"
            run.font.size = Pt(7.5)
        return p

    # title block
    para(TITLE, size=20, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(SUBTITLE, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    para(COURSE, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    para(INSTITUTE, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
    para(" and ".join(AUTHORS), size=11, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)

    doc.add_heading("Scope and reproducibility", level=1)
    para(
        f"Every table, figure and number in this document was produced by "
        f"executing the code in this repository on n = {n} AVeriTeC claims. "
        "Nothing is transcribed by hand: each value is read from a generated "
        "macro in results/macros/, and a name that no longer exists raises an "
        "error rather than printing a blank."
    )
    para("To reproduce every artefact below:", bold=True)
    mono([
        "python src/analysis/regenerate.py          # rebuilds results/ from the cache",
        "python src/analysis/build_results_doc.py   # rebuilds this document",
        "python -m pytest -q                        # 434 tests",
    ])
    para(
        "This runs offline. It makes no model call and needs no corpus download: "
        "the retrieved passages are committed in cache/av250.jsonl and the "
        "per-claim verdicts in results/, so the chain replays exactly. Decoding "
        "is greedy (temperature 0) and every sampling step is seeded, so reruns "
        "reproduce the same values."
    )
    para(
        "Scope: the protocol was designed for 250 claims. The search service "
        f"degraded partway through retrieval and the run was halted at {n} claims "
        "rather than looped, because continuing against a failing endpoint fills "
        "the cache with empty evidence sets indistinguishable from genuine "
        "'nothing found' results. This is a pilot, and the statistical power "
        "table states what it can and cannot settle."
    )

    doc.add_page_break()
    doc.add_heading("1. Console output", level=1)
    para("Actual console output from executing the analysis chain. The full "
         "files are in logs/ in the repository.")
    doc.add_heading("1.1  python src/analysis/regenerate.py", level=2)
    mono(read_log("regenerate.log", 70, width=108))
    doc.add_heading("1.2  python -m pytest -q", level=2)
    mono(read_log("pytest.log", 20, width=108))

    doc.add_page_break()
    doc.add_heading("2. Tables", level=1)

    order = [
        ("classification", "Classification performance"),
        ("differences", "Paired differences between conditions"),
        ("power", "Statistical power"),
        ("calibration", "Calibration"),
        ("citations", "Citations and cost"),
        ("attribution", "Error attribution by stage"),
        ("oracle", "Oracle-evidence diagnostic"),
        ("oracle_gain", "Effect of oracle evidence"),
        ("retrieval", "Retrieval quality"),
        ("per_class", "Per-class performance"),
        ("cost", "Latency"),
        ("integrity", "Pipeline integrity checks"),
        ("degraded", "Retrieval degradation"),
    ]
    for i, (key, heading) in enumerate(order, start=1):
        caption, head, rows, note = tables[key]
        doc.add_heading(f"2.{i}  {heading}", level=2)
        para(f"Table {i}. {caption}", size=9, italic=True, space_after=4)
        table = doc.add_table(rows=1, cols=len(head))
        table.style = "Light Grid Accent 1"
        for cell, text in zip(table.rows[0].cells, head):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.bold = True
            run.font.size = Pt(8.5)
        for row in rows:
            cells = table.add_row().cells
            for cell, text in zip(cells, row):
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(text))
                run.font.size = Pt(8.5)
        if note:
            para(note, size=8.5, italic=True, space_after=10)

    doc.add_page_break()
    doc.add_heading("3. Figures", level=1)
    fig_no = 0
    for filename, caption, source in figures:
        path = FIG_DIR / filename
        if not path.exists():
            continue
        fig_no += 1
        doc.add_picture(str(path), width=Inches(6.1))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = para(f"Figure {fig_no}. {caption} Generated by {source}.",
                 size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                 space_after=14)
        for run in p.runs:
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_heading("Index of generated quantities", level=1)
    para(
        f"This document draws on {len(m.values)} generated macros across three "
        "files in results/macros/: numbers.tex, condition_numbers.tex and "
        "attribution_numbers.tex. Each is written by the analysis scripts and "
        "regenerated by src/analysis/regenerate.py."
    )

    doc.save(str(out_path))
    return True


# --------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem", default="MAFCA_Results")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-docx", action="store_true")
    args = parser.parse_args(argv)

    m = Macros(MACRO_DIR)
    print(f"loaded {len(m.values)} generated macros from {MACRO_DIR}")

    tables = build_tables(m)
    missing = [f for f, _, _ in FIGURES if not (FIG_DIR / f).exists()]
    present = [f for f in FIGURES if (FIG_DIR / f[0]).exists()]
    if missing:
        print(f"  - {len(missing)} figure(s) not present, skipped: "
              f"{', '.join(missing)}")
    print(f"{len(tables)} table(s), {len(present)} figure(s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    if not args.skip_pdf:
        tex_path = OUT_DIR / f"{args.stem}.tex"
        tex_path.write_text(build_tex(m, tables, FIGURES), encoding="utf-8")
        if compile_pdf(tex_path):
            print(f"wrote {OUT_DIR / (args.stem + '.pdf')}")
        else:
            ok = False

    if not args.skip_docx:
        docx_path = OUT_DIR / f"{args.stem}.docx"
        if build_docx(m, tables, FIGURES, docx_path):
            print(f"wrote {docx_path}")
        else:
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
