"""Figures for evidence grounding, calibration and the aggregation ablation.

These cover the three claims the original results section could not support
from its own output:

* **Retrieval** (RQ2) -- Recall@k and Precision@k against annotated gold
  evidence URLs, at exact-URL and registered-domain granularity.
* **Calibration** -- a reliability diagram. The overconfidence gap reported
  in the original results is a signed mean, and a signed mean cancels: a
  system that is wildly overconfident on half its predictions and wildly
  underconfident on the other half scores a gap near zero. The reliability
  diagram and Expected Calibration Error show what the gap hides.
* **Ablation** -- accuracy, macro-F1 and ECE across aggregation policies
  replayed over the same fixed audits.

    python src/analysis/grounding_figures.py --results results/averitec_n60_v2.jsonl --tag av

Writes ``results/figures/<tag>_{retrieval,reliability,ablation}.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ablation import (  # noqa: E402
    apply_policy,
    card_for,
    make_deterministic,
    policy_as_run,
)

from fact_checker.metrics import calibration_bins  # noqa: E402
from fact_checker.retrieval_metrics import score_retrieval  # noqa: E402

FIG_DIR = Path("results/figures")

CRIMSON = "#A51C30"
NAVY = "#1E2761"
GREEN = "#2E7D32"
GREY = "#5B6472"


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _frame(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)


def retrieval_figure(records: Sequence[dict], tag: str) -> bool:
    """Recall@k and Precision@k at both matching granularities."""
    scored = [
        r
        for r in records
        if (r.get("multi") or {}).get("retrieved_urls") is not None and r.get("gold_urls")
    ]
    if not scored:
        return False

    retrieved = [r["multi"]["retrieved_urls"] for r in scored]
    gold = [r["gold_urls"] for r in scored]
    exact = score_retrieval(retrieved, gold, domain=False)
    domain = score_retrieval(retrieved, gold, domain=True)

    groups = ["Recall@k", "Precision@k", "Hit rate"]
    series = {
        "Exact canonical URL": [
            100 * exact.recall_at_k, 100 * exact.precision_at_k, 100 * exact.hit_rate
        ],
        "Registered domain": [
            100 * domain.recall_at_k, 100 * domain.precision_at_k, 100 * domain.hit_rate
        ],
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    width = 0.34
    for offset, (label, values), colour in zip(
        (-width / 2, width / 2), series.items(), (NAVY, CRIMSON)
    ):
        bars = ax.bar(
            [i + offset for i in range(len(groups))], values, width, label=label, color=colour
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.4,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=8.5,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Percent", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Evidence retrieval against gold URLs (n = {exact.n_claims})", fontsize=10.5
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _frame(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_retrieval.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _reliability_series(records: Sequence[dict], key: str) -> tuple[List[float], List[float], List[int]]:
    correct = [r[key]["verdict"] == r["gold"] for r in records]
    confidence = [float(r[key].get("confidence", 0.0)) for r in records]
    bins = calibration_bins(correct, confidence, bins=5)
    xs = [(b.lower + b.upper) / 2 for b in bins if b.count]
    ys = [b.accuracy for b in bins if b.count]
    ns = [b.count for b in bins if b.count]
    return xs, ys, ns


def reliability_figure(records: Sequence[dict], tag: str) -> bool:
    """Reliability diagram: stated confidence against achieved accuracy.

    A perfectly calibrated system lies on the diagonal. Bubble area shows how
    many predictions landed in each bin, because a bin holding two claims
    says much less than one holding twenty.
    """
    # Every claim, as in ablation.py: a claim with no retrieved
    # evidence has an empty audit, which aggregates to
    # NOT_ENOUGH_INFO at the no-evidence confidence -- what the run
    # emitted. Filtering them here put this figure on n = 88 while
    # the table beside it says 90.
    usable = list(records)
    if not any((r.get("multi") or {}).get("audit") for r in usable):
        return False

    conditions = [
        ("Single-LLM baseline", [dict(r, multi=r["baseline"]) for r in usable], NAVY, "o"),
        ("MAFCA, agent-reported", apply_policy(usable, policy_as_run), CRIMSON, "s"),
        (
            "MAFCA, computed",
            apply_policy(usable, make_deterministic(1.0, gate=True)),
            GREEN,
            "^",
        ),
    ]

    fig, axes = plt.subplots(
        1, len(conditions), figsize=(2.55 * len(conditions) + 0.6, 3.0),
        sharex=True, sharey=True,
    )
    if len(conditions) == 1:
        axes = [axes]

    for ax, (label, rows, colour, marker) in zip(axes, conditions):
        # Diagonal first and in a visible weight: it is the reference
        # the whole panel is read against.
        ax.plot([0, 1], [0, 1], linestyle="--", color="#9AA0A6",
                linewidth=1.1, zorder=1)
        ax.set_title(label, fontsize=8.2, color=colour, pad=6)
        if not rows:
            continue
        xs, ys, ns = _reliability_series(rows, "multi")
        if not xs:
            continue
        ax.plot(xs, ys, color=colour, linewidth=1.3, alpha=0.85, zorder=2)
        ax.scatter(xs, ys, s=30, color=colour, marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.7)
        # The count per bin, which bubble area encoded less precisely.
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{n}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6.4,
                        color=GREY)
        card = card_for(rows, "multi")
        ax.text(0.04, 0.93, f"ECE {card.ece:.3f}", transform=ax.transAxes,
                fontsize=7.0, color=GREY, va="top")

    for ax in axes:
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Stated confidence", fontsize=8.2)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7.2)
        ax.grid(color="#EEEEEE", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Observed accuracy", fontsize=8.2)
    fig.text(0.5, -0.04,
             "dashed = perfect calibration   ·   labels = claims in bin",
             ha="center", fontsize=7.0, color=GREY, style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def ablation_figure(records: Sequence[dict], tag: str, betas: Sequence[float]) -> bool:
    """Accuracy, macro-F1 and ECE across aggregation policies."""
    # All claims, matching ablation.py and Table 13. A claim with no
    # retrieved evidence has an empty audit, which aggregates to
    # NOT_ENOUGH_INFO at the no-evidence confidence; filtering it
    # put this figure on n = 88 beside a table saying 90.
    usable = list(records)
    if not usable:
        return False

    labels = ["agent-\nreported"] + [f"computed\nβ = {b:g}" for b in betas]
    conditions = [apply_policy(usable, policy_as_run)] + [
        apply_policy(usable, make_deterministic(b, gate=True)) for b in betas
    ]
    cards = [card_for(rows, "multi") for rows in conditions]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    width = 0.36
    x = range(len(labels))
    for offset, (name, values), colour in zip(
        (-width / 2, width / 2),
        (
            ("Accuracy", [100 * c.accuracy for c in cards]),
            ("Macro-F1", [100 * c.macro_f1 for c in cards]),
        ),
        (NAVY, CRIMSON),
    ):
        bars = left.bar([i + offset for i in x], values, width, label=name, color=colour)
        for bar in bars:
            left.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8,
            )
    left.set_xticks(list(x))
    left.set_xticklabels(labels, fontsize=8.5)
    left.set_ylabel("Percent", fontsize=10)
    left.set_ylim(0, 100)
    left.set_title("Classification", fontsize=10.5)
    left.legend(frameon=False, fontsize=8.5, loc="upper left")
    _frame(left)

    eces = [c.ece for c in cards]
    bars = right.bar(list(x), eces, 0.52, color=GREEN)
    for bar in bars:
        right.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
        )
    right.set_xticks(list(x))
    right.set_xticklabels(labels, fontsize=8.5)
    right.set_ylabel("Expected Calibration Error", fontsize=10)
    right.set_ylim(0, max(eces) * 1.3 if eces else 1.0)
    right.set_title("Calibration (lower is better)", fontsize=10.5)
    _frame(right)

    fig.suptitle(
        f"Aggregation policies replayed over the same audits (n = {len(usable)})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{tag}_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tag", default="av")
    parser.add_argument("--betas", type=float, nargs="*", default=[0.0, 1.0, 2.0])
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"No results file at {args.results}.", file=sys.stderr)
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = load(args.results)

    written: Dict[str, bool] = {
        "retrieval": retrieval_figure(records, args.tag),
        "reliability": reliability_figure(records, args.tag),
        "ablation": ablation_figure(records, args.tag, args.betas),
    }
    for name, ok in written.items():
        state = f"{FIG_DIR}/{args.tag}_{name}.png" if ok else "skipped (no stored trace)"
        print(f"  {name:<12} {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
