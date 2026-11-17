"""Chart for the poster, showing the same conditions as the poster's table.

    python src/analysis/poster_figures.py

The paper's accuracy figure plots the single LLM against both MAFCA
configurations, because the paper discusses the agent-reported variant
as an ablation. The poster's table lists the single LLM, single-call
RAG and MAFCA, and does not discuss the ablation at all, so reusing
the paper's figure put a series in the legend that nothing on the
poster explained and invited the third bar to be read as MAFCA.

This draws the table's conditions and nothing else, at font sizes that
survive A1 printing and a few metres of viewing distance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/analysis/..)

NAVY = "#1E2761"
CRIMSON = "#A51C30"
GREY = "#5B6472"
SOMAIYA_RED = "#B42133"
SOMAIYA_BLUE = "#01458E"

DECIDABLE = ("SUPPORTS", "REFUTES")


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(rows: Sequence[dict], decidable: bool = False) -> float:
    use = [r for r in rows if not decidable or r["gold"] in DECIDABLE]
    if not use:
        return 0.0
    return 100 * sum(1 for r in use if r["verdict"] == r["gold"]) / len(use)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "results" / "av250")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "figures" / "poster_accuracy.png"
    )
    args = parser.parse_args(argv)

    # Same three conditions the poster's table lists, in the same order.
    series = [
        ("B0: single LLM", "B0", SOMAIYA_BLUE),
        ("B1: single-call RAG", "B1", GREY),
        ("MAFCA", "B4", SOMAIYA_RED),
    ]

    missing = [c for _, c, _ in series if not (args.dir / f"{c}.jsonl").exists()]
    if missing:
        print(f"missing conditions: {missing}", file=sys.stderr)
        return 1

    data: Dict[str, List[dict]] = {c: load(args.dir / f"{c}.jsonl") for _, c, _ in series}
    n = len(data["B0"])

    groups = ["Overall", "Decidable\n(SUPPORTS / REFUTES)"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    width = 0.26
    x = range(len(groups))
    for i, (label, cond, colour) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        values = [accuracy(data[cond]), accuracy(data[cond], decidable=True)]
        bars = ax.bar([p + offset for p in x], values, width,
                      label=label, color=colour)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.6,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=15, fontweight="bold")

    # The majority-class floor, which beats every system on the
    # decidable subset. Drawn as a reference line because it is a floor
    # rather than a condition under test.
    if (args.dir / "MAJORITY.jsonl").exists():
        maj = load(args.dir / "MAJORITY.jsonl")
        floor = accuracy(maj, decidable=True)
        ax.plot([1 - 0.42, 1 + 0.42], [floor, floor], linestyle="--",
                linewidth=2.4, color="#222222", zorder=5)
        ax.text(1, floor + 1.4, f"majority-class floor {floor:.1f}",
                va="bottom", ha="center", fontsize=13, color="#222222")

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=17)
    ax.set_ylim(0, 72)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(frameon=False, fontsize=15, loc="upper left", ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.9)
    ax.set_axisbelow(True)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.out}  (n = {n} claims)")
    for label, cond, _ in series:
        print(f"  {label:22s} overall {accuracy(data[cond]):5.1f}   "
              f"decidable {accuracy(data[cond], decidable=True):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
