"""Figures for the pilot benchmark section of the paper.

Every value plotted here is read from ``results/fever_n30.jsonl`` -- the raw
per-claim output of an actual benchmark run. Nothing is hardcoded, so the
figures cannot drift away from the run they describe.

    python src/analysis/pilot_figures.py [--results results/fever_n30.jsonl]

Produces:
  results/figures/fig5_pilot_label_groups.png  - accuracy by gold-label group
  results/figures/fig6_reliability.png         - reliability diagram
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fact_checker.metrics import calibration_bins  # noqa: E402

FIG_DIR = Path("results/figures")
DEFAULT_RESULTS = Path("results/fever_n30.jsonl")

CRIMSON = "#A51C30"
NAVY = "#1E2761"
GREY = "#5B6472"

DECIDABLE = ("SUPPORTS", "REFUTES")


def load_rows(path: Path) -> List[dict]:
    """Read the per-claim benchmark records."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(rows: Sequence[dict], system: str) -> float:
    """Fraction of ``rows`` on which ``system`` matched the gold label."""
    if not rows:
        return 0.0
    return 100.0 * sum(r[system]["verdict"] == r["gold"] for r in rows) / len(rows)


def label_groups(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    """Split rows into the blended, decidable, and NEI-only reporting groups."""
    return {
        "blended": list(rows),
        "decidable": [r for r in rows if r["gold"] in DECIDABLE],
        "nei": [r for r in rows if r["gold"] == "NOT_ENOUGH_INFO"],
    }


def fig_label_groups(rows: Sequence[dict]) -> None:
    """Accuracy by gold-label group, for both systems."""
    groups = label_groups(rows)
    order = ["blended", "decidable", "nei"]
    labels = [
        f"Overall (blended)\nn = {len(groups['blended'])}",
        f"Decidable (SUPPORTS / REFUTES)\nn = {len(groups['decidable'])}",
        f"NOT_ENOUGH_INFO only\nn = {len(groups['nei'])}",
    ]
    multi = [accuracy(groups[g], "multi") for g in order]
    base = [accuracy(groups[g], "baseline") for g in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = range(len(order))
    width = 0.34
    bars = (
        ax.bar([i - width / 2 for i in x], multi, width,
               label="Multi-agent (grounded)", color=CRIMSON),
        ax.bar([i + width / 2 for i in x], base, width,
               label="Baseline (ungrounded)", color=NAVY),
    )
    for group in bars:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 118)
    ax.axhline(33.3, color=GREY, linewidth=0.9, linestyle="--")
    ax.text(len(order) - 0.52, 35.0, "chance (33.3%)", fontsize=8,
            color=GREY, ha="right")
    ax.axvline(0.5, color="#BBBBBB", linewidth=0.9, linestyle=":")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_pilot_label_groups.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(rows: Sequence[dict], bins: int = 10) -> None:
    """Reliability diagram: bin accuracy against bin mean confidence."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=1.0, linestyle="--",
            label="perfect calibration")

    for system, colour, name in (
        ("multi", CRIMSON, "Multi-agent (grounded)"),
        ("baseline", NAVY, "Baseline (ungrounded)"),
    ):
        correct = [r[system]["verdict"] == r["gold"] for r in rows]
        confidence = [float(r[system]["confidence"]) for r in rows]
        populated = [b for b in calibration_bins(correct, confidence, bins) if b.count]
        ax.plot(
            [b.mean_confidence for b in populated],
            [b.accuracy for b in populated],
            marker="o", markersize=5, linewidth=1.5, color=colour, label=name,
        )
        # Marker area scales with how many predictions landed in the bin.
        ax.scatter(
            [b.mean_confidence for b in populated],
            [b.accuracy for b in populated],
            s=[18 * b.count for b in populated], color=colour, alpha=0.22,
        )

    ax.set_xlabel("Mean confidence in bin", fontsize=10)
    ax.set_ylabel("Accuracy in bin", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.04, "below the diagonal = overconfident",
            fontsize=8, color=GREY, ha="right", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Per-claim benchmark JSONL to plot.")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"No results file at {args.results}. Run evaluate.py first.")
        return 1

    rows = load_rows(args.results)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_label_groups(rows)
    fig_reliability(rows)
    print(f"Wrote pilot figures for n={len(rows)} to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
