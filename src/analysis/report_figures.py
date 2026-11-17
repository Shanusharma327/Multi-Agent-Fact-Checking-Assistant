"""Figures used only by the final project report (REPORT .docx).

    python src/analysis/report_figures.py

Produces:
  results/figures/fig_architecture.png   - system architecture block diagram
  results/figures/fig_results.png        - accuracy by label group
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_DIR = Path("results/figures")

CRIMSON = "#A51C30"
NAVY = "#1E2761"
GREEN = "#2E7D32"
ORANGE = "#EF6C00"
GREY = "#5B6472"
LIGHT = "#F2F5FA"
WARM = "#FDF3F4"


def _box(ax, x, y, w, h, title, subtitle, edge, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=edge, zorder=3)
    ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
            fontsize=8.2, color=GREY, zorder=3, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=13,
            linewidth=1.4, color=color, zorder=1,
        )
    )


def architecture() -> None:
    """Block diagram of the three-agent pipeline and the control path."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Input
    _box(ax, 3.6, 5.45, 2.8, 0.62, "USER CLAIM", "Free-text assertion", NAVY, LIGHT)

    # Multi-agent column
    _arrow(ax, 4.3, 5.43, 2.55, 5.05)
    _box(ax, 0.55, 4.05, 4.0, 0.95, "AGENT 1  ·  Evidence Scout",
         "Generates 2 neutral queries → web search\nReturns snippets as doc_1 … doc_n", CRIMSON, LIGHT)
    _arrow(ax, 2.55, 4.03, 2.55, 3.35)
    _box(ax, 0.55, 2.35, 4.0, 0.95, "AGENT 2  ·  Alignment Auditor",
         "Closed-world NLI on each snippet alone\nSUPPORTS / REFUTES / NOT_ENOUGH_INFO", NAVY, LIGHT)
    _arrow(ax, 2.55, 2.33, 2.55, 1.65)
    _box(ax, 0.55, 0.65, 4.0, 0.95, "AGENT 3  ·  Synthesis Adjudicator",
         "Weighs alignments → unified verdict\nCalibrated confidence + cited sources", GREEN, LIGHT)

    # Baseline column
    _arrow(ax, 5.7, 5.43, 7.45, 5.05)
    _box(ax, 5.45, 4.05, 4.0, 0.95, "BASELINE  ·  Single LLM (Control)",
         "Zero-shot, no retrieval, no sub-agents\nParametric knowledge only", ORANGE, WARM)
    _arrow(ax, 7.45, 4.03, 7.45, 1.65)

    # Shared schema
    _box(ax, 5.45, 0.65, 4.0, 0.95, "IDENTICAL OUTPUT SCHEMA",
         "verdict · confidence · reasoning · citations\nEnables a strict 1:1 comparison", NAVY, WARM)

    _arrow(ax, 4.57, 1.12, 5.43, 1.12, color=NAVY, style="<|-|>")
    ax.text(5.0, 1.32, "compare", ha="center", va="bottom",
            fontsize=8.5, style="italic", color=NAVY)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def results() -> None:
    """Accuracy by label group -- the inversion of the blended headline."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    groups = ["Overall\n(blended)", "Decidable\n(SUPPORTS / REFUTES)", "NOT_ENOUGH_INFO\nonly"]
    multi = [65.0, 46.2, 100.0]
    base = [55.0, 69.2, 28.6]
    x = range(len(groups))
    width = 0.34

    b1 = ax.bar([i - width / 2 for i in x], multi, width,
                label="Multi-agent (grounded)", color=CRIMSON)
    b2 = ax.bar([i + width / 2 for i in x], base, width,
                label="Baseline (ungrounded)", color=NAVY)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.9, linestyle=":")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def untitled_variants() -> None:
    """Caption-free copies of the two shared figures, for the .docx report.

    The document supplies numbered captions (Figure 4.1, Figure 4.2), so the
    baked-in "Figure 1:" / "Figure 3:" titles would contradict them.
    """
    from dataset_stats import (  # noqa: E402  - same package directory
        compute_averitec, compute_fever, fig_fever_evidence, fig_label_distribution,
    )
    from fact_checker.datasets.averitec import load_averitec
    from fact_checker.datasets.fever import load_fever

    fever = compute_fever(load_fever("dev", "data/fever", download=False))
    averitec_claims, report = load_averitec("dev", "data/averitec", download=False)
    averitec = compute_averitec(averitec_claims, report)

    fig_label_distribution({"fever": fever, "averitec": averitec},
                           titled=False, suffix="_report")
    fig_fever_evidence(fever, titled=False, suffix="_report")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    results()
    untitled_variants()
    print(f"Wrote report figures to {FIG_DIR}/ "
          "(architecture, results, and caption-free dataset variants)")
