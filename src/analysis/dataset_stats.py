"""Descriptive analysis of the benchmark corpora used in this project.

Covers both FEVER (comparability) and AVeriTeC (validity), and produces the
statistics tables and figures cited in the project report (now at
_archive/deliverables/REPORT.md §7). Everything is computed
from the locally cached files -- no numbers are hard-coded.

    python src/analysis/dataset_stats.py                  # both, if cached
    python src/analysis/dataset_stats.py --download       # fetch anything missing
    python src/analysis/dataset_stats.py --dataset fever  # one only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Allow running as `python src/analysis/dataset_stats.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from fact_checker.datasets.averitec import load_averitec  # noqa: E402
from fact_checker.datasets.base import BenchmarkClaim  # noqa: E402
from fact_checker.datasets.fever import load_fever  # noqa: E402

FIG_DIR = Path("results/figures")
OUT_JSON = Path("analysis/dataset_stats.json")

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
SHORT = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES", "NOT_ENOUGH_INFO": "NEI"}

# Colour-blind-safe, consistent across every figure.
COLORS = {"SUPPORTS": "#2E7D32", "REFUTES": "#C62828", "NOT_ENOUGH_INFO": "#EF6C00"}
DATASET_COLORS = {"fever": "#37474F", "averitec": "#5E35B1"}
DATASET_TITLES = {"fever": "FEVER (dev)", "averitec": "AVeriTeC (dev)"}
SHORT_DATASET = {"fever": "FEVER", "averitec": "AVeriTeC"}
GRID = {"color": "#DDDDDD", "linewidth": 0.7}


def _style(ax) -> None:
    """Apply one consistent axis style to every figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _length_stats(values: Sequence[int]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def compute_common(claims: Sequence[BenchmarkClaim]) -> Dict:
    """Statistics that apply to any corpus normalised onto our label set."""
    counts = Counter(c.label for c in claims)
    total = len(claims)
    words: Dict[str, List[int]] = {label: [] for label in LABELS}
    for claim in claims:
        words[claim.label].append(len(claim.claim.split()))
    all_words = [n for bucket in words.values() for n in bucket]

    return {
        "total": total,
        "label_counts": {label: counts[label] for label in LABELS},
        "label_pct": {label: 100.0 * counts[label] / total for label in LABELS},
        "claim_words": _length_stats(all_words),
        "claim_chars": _length_stats([len(c.claim) for c in claims]),
        "words_by_label": {
            label: _length_stats(words[label]) if words[label] else {}
            for label in LABELS
        },
        "_raw_words": words,
        "_all_words": all_words,
    }


def compute_fever(claims) -> Dict:
    """FEVER-specific: gold Wikipedia page annotations and the verifiable flag."""
    stats = compute_common(claims)
    pages: Dict[str, List[int]] = {label: [] for label in LABELS}
    for claim in claims:
        pages[claim.label].append(len(claim.gold_pages))

    stats["verifiable"] = dict(Counter(c.verifiable for c in claims))
    stats["gold_pages_by_label"] = {
        label: {
            "mean": statistics.mean(pages[label]),
            "max": max(pages[label]),
            "zero_pct": 100.0 * sum(1 for n in pages[label] if n == 0) / len(pages[label]),
        }
        for label in LABELS
    }
    stats["gold_pages_overall_mean"] = statistics.mean(
        [n for bucket in pages.values() for n in bucket]
    )
    stats["_raw_pages"] = pages
    return stats


def compute_averitec(claims, load_report: Dict) -> Dict:
    """AVeriTeC-specific: provenance, claim types, and evidence QA counts."""
    stats = compute_common(claims)
    stats["load_report"] = load_report
    stats["claim_types"] = dict(
        Counter(t for c in claims for t in c.claim_types).most_common()
    )
    stats["reporting_sources"] = dict(
        Counter(c.reporting_source for c in claims if c.reporting_source).most_common()
    )
    stats["claim_years"] = dict(
        Counter(
            c.claim_date.split("-")[-1]
            for c in claims
            if c.claim_date and len(c.claim_date.split("-")) == 3
        )
    )
    stats["questions_per_claim"] = _length_stats([c.n_questions for c in claims])
    stats["answers_per_claim"] = _length_stats([c.n_answers for c in claims])
    stats["_raw_questions"] = [c.n_questions for c in claims]
    return stats


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_label_distribution(by_dataset: Dict[str, Dict], titled: bool = True,
                           suffix: str = "") -> None:
    """Figure 1: class balance, one panel per corpus."""
    names = list(by_dataset)
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 3.8))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        stats = by_dataset[name]
        counts = [stats["label_counts"][label] for label in LABELS]
        bars = ax.bar(
            [SHORT[label] for label in LABELS],
            counts,
            color=[COLORS[label] for label in LABELS],
            width=0.62,
        )
        for bar, count in zip(bars, counts):
            pct = 100.0 * count / stats["total"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.015,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_ylabel("Number of claims")
        ax.set_title(
            f"{DATASET_TITLES[name]} — n = {stats['total']:,}", fontsize=10
        )
        ax.set_ylim(0, max(counts) * 1.25)
        _style(ax)

    if titled:
        # Omitted for the report, where the document supplies the caption.
        fig.suptitle(
            "Figure 1: Label distribution — balanced FEVER vs. imbalanced AVeriTeC",
            fontsize=11,
            y=1.02,
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig1_label_distribution{suffix}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_claim_length(by_dataset: Dict[str, Dict]) -> None:
    """Figure 2: claim length, overlaid and compared across corpora."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8))

    bins = range(0, 56, 2)
    for name, stats in by_dataset.items():
        ax1.hist(
            stats["_all_words"],
            bins=bins,
            density=True,
            alpha=0.55,
            label=f"{DATASET_TITLES[name]} (mean {stats['claim_words']['mean']:.1f})",
            color=DATASET_COLORS[name],
            edgecolor="white",
            linewidth=0.5,
        )
    ax1.set_xlabel("Claim length (words)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Claim length, normalised", fontsize=10)
    ax1.legend(fontsize=8.5, frameon=False)
    _style(ax1)

    data, ticks, colors = [], [], []
    for name, stats in by_dataset.items():
        for label in LABELS:
            if stats["_raw_words"][label]:
                data.append(stats["_raw_words"][label])
                ticks.append(f"{SHORT[label]}\n{SHORT_DATASET[name]}")
                colors.append(COLORS[label])
    parts = ax2.boxplot(
        data, tick_labels=ticks, patch_artist=True, showfliers=False, widths=0.55
    )
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in parts["medians"]:
        median.set_color("black")
    ax2.set_ylabel("Claim length (words)")
    ax2.set_title("(b) By gold label and corpus", fontsize=10)
    ax2.tick_params(axis="x", labelsize=8)
    _style(ax2)

    fig.suptitle(
        "Figure 2: AVeriTeC claims are twice as long and self-contained",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_claim_length.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_fever_evidence(stats: Dict, titled: bool = True, suffix: str = "") -> None:
    """Figure 3: FEVER gold evidence pages per claim, by label."""
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    width = 0.26
    buckets = [0, 1, 2, 3]
    labels_x = ["0 pages", "1 page", "2 pages", "3+ pages"]

    for offset, label in zip((-width, 0.0, width), LABELS):
        pages = stats["_raw_pages"][label]
        total = len(pages)
        heights = [
            100.0 * sum(1 for n in pages if n == b) / total
            if b < 3
            else 100.0 * sum(1 for n in pages if n >= 3) / total
            for b in buckets
        ]
        ax.bar(
            [i + offset for i in range(len(buckets))],
            heights,
            width=width,
            label=SHORT[label],
            color=COLORS[label],
        )

    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("% of claims with that label")
    if titled:
        ax.set_title(
            "Figure 3: FEVER gold evidence pages — every NEI claim has none",
            fontsize=10.5,
        )
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig3_fever_evidence_pages{suffix}.png", dpi=160)
    plt.close(fig)


def fig_averitec_composition(stats: Dict) -> None:
    """Figure 4: what AVeriTeC actually contains — provenance and claim types."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    types = list(stats["claim_types"].items())[:6][::-1]
    ax1.barh(
        [t[0].replace(" Claim", "") for t in types],
        [t[1] for t in types],
        color=DATASET_COLORS["averitec"],
    )
    ax1.set_xlabel("Number of claims")
    ax1.set_title("(a) Claim types", fontsize=10)
    ax1.grid(axis="x", **GRID)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    sources = list(stats["reporting_sources"].items())[:6][::-1]
    ax2.barh(
        [s[0][:30] for s in sources],
        [s[1] for s in sources],
        color="#00796B",
    )
    ax2.set_xlabel("Number of claims")
    ax2.set_title("(b) Reporting source", fontsize=10)
    ax2.grid(axis="x", **GRID)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    questions = stats["_raw_questions"]
    ax3.hist(
        questions,
        bins=range(0, max(questions) + 2),
        color="#EF6C00",
        edgecolor="white",
    )
    ax3.axvline(
        stats["questions_per_claim"]["mean"],
        color="#C62828",
        linestyle="--",
        linewidth=1.4,
        label=f"mean = {stats['questions_per_claim']['mean']:.2f}",
    )
    ax3.set_xlabel("Evidence questions per claim")
    ax3.set_ylabel("Number of claims")
    ax3.set_title("(c) Annotated evidence", fontsize=10)
    ax3.legend(fontsize=8.5, frameon=False)
    _style(ax3)

    fig.suptitle(
        "Figure 4: AVeriTeC composition — real circulating misinformation",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_averitec_composition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(name: str, stats: Dict) -> None:
    """Console tables, mirroring what the archived report §7 cites."""
    print(f"\n{'=' * 72}")
    print(f"{DATASET_TITLES[name]}: {stats['total']:,} usable claims")
    print("=" * 72)

    if "load_report" in stats:
        report = stats["load_report"]
        print(
            f"  loaded {report['kept']}/{report['total']} records "
            f"({report.get('conflicting', 0)} excluded as Conflicting, "
            f"{report.get('unusable', 0)} unusable)"
        )

    header = f"{'Label':<18}{'Count':>9}{'Share':>9}{'Mean words':>12}"
    if "gold_pages_by_label" in stats:
        header += f"{'Mean pages':>12}"
    print(header)
    for label in LABELS:
        row = (
            f"{label:<18}{stats['label_counts'][label]:>9,}"
            f"{stats['label_pct'][label]:>8.1f}%"
            f"{stats['words_by_label'][label].get('mean', 0):>12.2f}"
        )
        if "gold_pages_by_label" in stats:
            row += f"{stats['gold_pages_by_label'][label]['mean']:>12.2f}"
        print(row)

    words = stats["claim_words"]
    print(
        f"\n  claim length : mean {words['mean']:.2f}, median {words['median']:.1f}, "
        f"sd {words['stdev']:.2f}, range {words['min']:.0f}-{words['max']:.0f} words"
    )
    print(f"  mean chars   : {stats['claim_chars']['mean']:.1f}")

    if "verifiable" in stats:
        print(f"  verifiable   : {stats['verifiable']}")
    if "claim_types" in stats:
        top_types = list(stats["claim_types"].items())[:4]
        top_src = list(stats["reporting_sources"].items())[:4]
        print(f"  claim types  : {top_types}")
        print(f"  top sources  : {top_src}")
        print(f"  claim years  : {stats['claim_years']}")
        print(
            f"  evidence     : {stats['questions_per_claim']['mean']:.2f} questions, "
            f"{stats['answers_per_claim']['mean']:.2f} answers per claim"
        )


def print_comparison(by_dataset: Dict[str, Dict]) -> None:
    """Side-by-side contrast — the table the archived report §7.6 reproduces."""
    if len(by_dataset) < 2:
        return
    fever, averitec = by_dataset.get("fever"), by_dataset.get("averitec")
    if not (fever and averitec):
        return

    print(f"\n{'=' * 72}")
    print("FEVER vs AVeriTeC")
    print("=" * 72)
    print(f"{'Property':<28}{'FEVER':>18}{'AVeriTeC':>18}")
    rows = [
        ("usable claims", f"{fever['total']:,}", f"{averitec['total']:,}"),
        (
            "mean claim words",
            f"{fever['claim_words']['mean']:.1f}",
            f"{averitec['claim_words']['mean']:.1f}",
        ),
        (
            "median claim words",
            f"{fever['claim_words']['median']:.1f}",
            f"{averitec['claim_words']['median']:.1f}",
        ),
        (
            "mean claim chars",
            f"{fever['claim_chars']['mean']:.1f}",
            f"{averitec['claim_chars']['mean']:.1f}",
        ),
    ]
    for label in LABELS:
        rows.append(
            (
                f"{label} share",
                f"{fever['label_pct'][label]:.1f}%",
                f"{averitec['label_pct'][label]:.1f}%",
            )
        )
    for name, a, b in rows:
        print(f"{name:<28}{a:>18}{b:>18}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["fever", "averitec", "both"],
        default="both",
        help="Which corpus to analyse.",
    )
    parser.add_argument(
        "--download", action="store_true", help="Fetch a split if not cached."
    )
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wanted = ["fever", "averitec"] if args.dataset == "both" else [args.dataset]
    by_dataset: Dict[str, Dict] = {}

    for name in wanted:
        try:
            if name == "fever":
                claims = load_fever("dev", "data/fever", download=args.download)
                by_dataset[name] = compute_fever(claims)
            else:
                claims, report = load_averitec(
                    "dev", "data/averitec", download=args.download
                )
                by_dataset[name] = compute_averitec(claims, report)
        except Exception as exc:  # noqa: BLE001 - a missing cache is normal.
            print(
                f"  skipping {name}: {exc}\n"
                f"  (run with --download, or: python evaluate.py --dataset {name} --download)"
            )

    if not by_dataset:
        print("No corpora available to analyse.", file=sys.stderr)
        return 1

    fig_label_distribution(by_dataset)
    fig_claim_length(by_dataset)
    if "fever" in by_dataset:
        fig_fever_evidence(by_dataset["fever"])
    if "averitec" in by_dataset:
        fig_averitec_composition(by_dataset["averitec"])

    serialisable = {
        name: {k: v for k, v in stats.items() if not k.startswith("_")}
        for name, stats in by_dataset.items()
    }
    OUT_JSON.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    for name, stats in by_dataset.items():
        print_summary(name, stats)
    print_comparison(by_dataset)

    print(f"\nFigures -> {FIG_DIR}/   Stats -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
