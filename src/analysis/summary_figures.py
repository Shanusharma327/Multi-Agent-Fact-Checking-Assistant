"""Summary figures for the proposed-methodology section.

    python src/analysis/summary_figures.py

Produces:
  results/figures/fig_flowchart.png   - input-to-output flow of the pipeline
  results/figures/fig_summary_bar.png - multi-metric grouped comparison,
                                         computed from the results files

The grouped bar chart reads the benchmark output rather than hardcoding
values, so it cannot drift away from the runs it describes.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from fact_checker.metrics import score_system  # noqa: E402

FIG_DIR = Path("results/figures")

CRIMSON = "#A51C30"
NAVY = "#1E2761"
GREEN = "#2E7D32"
ORANGE = "#EF6C00"
GREY = "#5B6472"
LIGHT = "#F2F5FA"

# One corpus, one run. FEVER was withdrawn from the paper and the
# 60-claim AVeriTeC pilot was superseded by the 90-claim run; this
# figure was still drawing both.
RUNS = (("results/averitec_n90_v2.jsonl", "AVeriTeC\n(n = 90)"),)


# ----------------------------------------------------------------------
# Flowchart
# ----------------------------------------------------------------------

def _stage(ax, y, title, detail, colour):
    ax.add_patch(FancyBboxPatch(
        (0.8, y), 6.4, 0.72,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.5, edgecolor=colour, facecolor=LIGHT, zorder=2))
    ax.text(4.0, y + 0.47, title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=colour, zorder=3)
    ax.text(4.0, y + 0.19, detail, ha="center", va="center",
            fontsize=7.6, color=GREY, zorder=3)


def _down(ax, y_from, y_to, label=""):
    ax.add_patch(FancyArrowPatch((4.0, y_from), (4.0, y_to),
                                 arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color=GREY, zorder=1))
    if label:
        ax.text(4.18, (y_from + y_to) / 2, label, ha="left", va="center",
                fontsize=7.2, style="italic", color=GREY)


def flowchart() -> None:
    """Input-to-output flow, one box per phase."""
    fig, ax = plt.subplots(figsize=(6.0, 8.6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    stages = [
        (8.4, "INPUT PHASE", "Claim text from the benchmark corpus", NAVY),
        (7.4, "PRE-PROCESSING PHASE",
         "Label mapping; 4th class excluded; claim text left verbatim", NAVY),
        (6.4, "AGENT 1  ·  EVIDENCE RETRIEVAL",
         "Two neutral queries → web search → leakage filter", CRIMSON),
        (5.4, "FEATURE / EVIDENCE REPRESENTATION",
         "De-duplicate, truncate to 700 chars, assign doc_1 … doc_n", CRIMSON),
        (4.4, "AGENT 2  ·  EVIDENCE VERIFICATION",
         "Closed-world NLI on each passage alone + quoted justification",
         NAVY),
        (3.4, "POST-CONDITION  ·  AUDIT RECONCILIATION",
         "Drop unknown ids; backfill NOT_ENOUGH_INFO; one label per passage",
         GREEN),
        (2.4, "AGENT 3  ·  DECISION PHASE",
         "Weigh audit by specificity → verdict + confidence", NAVY),
        (1.4, "POST-CONDITION  ·  CITATION VALIDATION",
         "Remove fabricated ids; recover inline refs; pin original claim",
         GREEN),
        (0.4, "OUTPUT PHASE",
         "Verdict · confidence κ · reasoning · validated citations", ORANGE),
    ]
    for y, title, detail, colour in stages:
        _stage(ax, y, title, detail, colour)
    for (y_hi, *_), (y_lo, *_) in zip(stages, stages[1:]):
        _down(ax, y_hi, y_lo + 0.72)

    # the empty-evidence short circuit
    ax.add_patch(FancyArrowPatch((7.2, 6.76), (7.75, 6.76),
                                 arrowstyle="-", linewidth=1.1, color=ORANGE))
    ax.add_patch(FancyArrowPatch((7.75, 6.76), (7.75, 2.76),
                                 arrowstyle="-", linewidth=1.1, color=ORANGE))
    ax.add_patch(FancyArrowPatch((7.75, 2.76), (7.2, 2.76),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.1, color=ORANGE))
    ax.text(7.86, 4.8, "no evidence →\nNOT_ENOUGH_INFO", rotation=90,
            ha="center", va="center", fontsize=6.8, color=ORANGE)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Multi-metric summary
# ----------------------------------------------------------------------

def _metrics(path: str) -> Dict[str, Dict[str, float]]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    gold = [r["gold"] for r in rows]
    out: Dict[str, Dict[str, float]] = {}
    for key in ("baseline", "multi"):
        card = score_system(key, gold, [r[key]["verdict"] for r in rows],
                            [float(r[key]["confidence"]) for r in rows])
        out[key] = {
            "Accuracy": 100 * card.accuracy,
            "Macro-P": 100 * statistics.mean(s.precision for s in card.per_label),
            "Macro-R": 100 * statistics.mean(s.recall for s in card.per_label),
            "Macro-F1": 100 * card.macro_f1,
        }
    return out


def summary_bar() -> None:
    """Grouped bars: four metrics, both systems, one panel per dataset."""
    names = ["Accuracy", "Macro-P", "Macro-R", "Macro-F1"]
    # squeeze=False keeps this an array even with a single corpus, so
    # the loop and the axes[0] labelling below work either way.
    width = 5.4 if len(RUNS) == 1 else 9.6
    fig, axes = plt.subplots(
        1, len(RUNS), figsize=(width, 3.9), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, (path, label) in zip(axes, RUNS):
        if not Path(path).exists():
            ax.set_visible(False)
            continue
        m = _metrics(path)
        x = range(len(names))
        width = 0.34
        for offset, key, colour, legend in (
            (-width / 2, "baseline", NAVY, "Existing (single-LLM)"),
            (width / 2, "multi", CRIMSON, "Proposed (multi-agent)"),
        ):
            bars = ax.bar([i + offset for i in x], [m[key][n] for n in names],
                          width, label=legend, color=colour)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{bar.get_height():.1f}", ha="center", va="bottom",
                        fontsize=7.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=8.6)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score (%)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Network / communication model
# ----------------------------------------------------------------------

def _node(ax, x, y, w, h, title, sub, colour, fill=LIGHT):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.4, edgecolor=colour, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=colour, zorder=3)
    ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
            fontsize=6.8, color=GREY, zorder=3)


def _link(ax, p, q, colour=GREY, dashed=False, label="", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=11, linewidth=1.2,
        color=colour, zorder=1,
        linestyle="--" if dashed else "-"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label,
                ha="center", va="bottom", fontsize=6.6, color=colour,
                style="italic")


def network_model() -> None:
    """Message-passing topology between agents and external services.

    Laid out as three columns -- search on the left, the agent chain in
    the middle, the inference backend on the right -- so that every
    external call is a short stub rather than a diagonal across the
    figure. The backend is one tall panel because all three agents and
    the baseline call the same endpoint.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    # Rows: one per agent, top to bottom.
    row1, row2, row3 = 4.60, 3.20, 1.80
    h = 0.85
    mid = lambda y: y + h / 2          # noqa: E731  vertical centre of a row

    # Left column: the one external service only Agent 1 talks to.
    _node(ax, 0.20, row1, 2.6, h, "WEB SEARCH SERVICE",
          "external, rate-limited", ORANGE, "#FDF3F4")

    # Middle column: the agent chain.
    _node(ax, 3.50, row1, 3.1, h, "AGENT 1 · Retrieval",
          "emits ScoutReport", CRIMSON)
    _node(ax, 3.50, row2, 3.1, h, "AGENT 2 · Verification",
          "emits AuditReport", NAVY)
    _node(ax, 3.50, row3, 3.1, h, "AGENT 3 · Verdict",
          "emits Adjudication", GREEN)

    # Right column: one endpoint every agent calls, drawn as a panel
    # spanning their full height so each stub stays horizontal.
    back_bottom, back_top = row3, row1 + h
    ax.add_patch(FancyBboxPatch(
        (7.40, back_bottom), 2.40, back_top - back_bottom,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.4, edgecolor=ORANGE, facecolor="#FDF3F4", zorder=2))
    ax.text(8.60, back_top - 0.34, "INFERENCE", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=ORANGE, zorder=3)
    ax.text(8.60, back_top - 0.70, "BACKEND", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color=ORANGE, zorder=3)
    ax.text(8.60, back_top - 1.06, "local or hosted LLM", ha="center",
            va="center", fontsize=6.8, color=GREY, zorder=3)

    # The control, directly beneath the endpoint it also calls.
    _node(ax, 7.40, 0.45, 2.40, h, "BASELINE",
          "emits Adjudication", ORANGE, "#FDF3F4")

    # Schema-validated messages down the chain.
    # Labels sit beside the arrow, not on it: the gap between rows is
    # short enough that a centred label overprints the line.
    _link(ax, (5.05, row1), (5.05, row2 + h), NAVY)
    ax.text(5.20, (row1 + row2 + h) / 2, "evidence list", ha="left",
            va="center", fontsize=6.6, color=NAVY, style="italic")
    _link(ax, (5.05, row2), (5.05, row3 + h), GREEN)
    ax.text(5.20, (row2 + row3 + h) / 2, "audit + tally", ha="left",
            va="center", fontsize=6.6, color=GREEN, style="italic")

    # External calls: horizontal stubs, one per row.
    _link(ax, (3.50, mid(row1)), (2.80, mid(row1)), ORANGE,
          dashed=True, style="<|-|>")
    for row in (row1, row2, row3):
        _link(ax, (6.60, mid(row)), (7.40, mid(row)), ORANGE,
              dashed=True, style="<|-|>")
    # ...and one short vertical stub for the baseline.
    _link(ax, (8.60, 1.30), (8.60, back_bottom), ORANGE,
          dashed=True, style="<|-|>")

    ax.text(5.0, 0.12, "solid = schema-validated inter-agent message   ·   "
                       "dashed = external service call",
            ha="center", va="center", fontsize=7.0, color=GREY, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    flowchart()
    summary_bar()
    network_model()
    print(f"Wrote fig_flowchart.png, fig_summary_bar.png and "
          f"fig_network.png to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
