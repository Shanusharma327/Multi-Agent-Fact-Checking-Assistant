"""Aggregation ablation by replay over stored per-passage audits.

Re-running the whole pipeline for each aggregation policy would confound the
comparison: live web search returns different documents from one hour to the
next, so two conditions run separately differ in their evidence as well as in
their aggregation. Replaying instead holds the retrieval *and* the
verification audit fixed and varies only the function that turns an audit
into a verdict. Whatever difference appears is then attributable to that
function alone.

Replay performs no inference, so it is free and instantaneous. It requires a
results file written by the current ``evaluate.py``, which stores the audit
alongside the verdict.

    python src/analysis/ablation.py --results results/averitec_det.jsonl

Writes an ablation table to ``results/tables/<tag>_ablation.tex`` and prints
the same numbers to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.aggregation import (  # noqa: E402
    aggregate,
    consistent_citations,
    supporting_ids,
)
from fact_checker.metrics import Scorecard, score_system  # noqa: E402
from fact_checker.retrieval_metrics import score_citations  # noqa: E402
from fact_checker.schemas import Alignment, AuditReport  # noqa: E402

TAB_DIR = Path("results/tables")


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rebuild_audit(block: dict) -> AuditReport:
    """Reconstruct the per-passage audit stored in a result record."""
    return AuditReport(
        alignments=[
            Alignment(
                source_id=str(entry.get("source_id", "")),
                verdict=entry.get("verdict", "NOT_ENOUGH_INFO"),
                justification=str(entry.get("justification", "")),
            )
            for entry in (block.get("audit") or [])
            if entry.get("source_id")
        ]
    )


# A policy maps a stored result block to (verdict, confidence, citations).
Policy = Callable[[dict], Tuple[str, float, List[str]]]


def policy_as_run(block: dict) -> Tuple[str, float, List[str]]:
    """Whatever the run itself produced. The reference condition."""
    return (
        block["verdict"],
        float(block.get("confidence", 0.0)),
        list(block.get("cited_sources") or []),
    )


def make_deterministic(beta: float, *, gate: bool) -> Policy:
    """Build a replay policy applying Eqs. (18)-(19) at a given ``beta``."""

    def policy(block: dict) -> Tuple[str, float, List[str]]:
        audit = rebuild_audit(block)
        trace = aggregate(audit, beta=beta)
        valid = set(block.get("evidence_ids") or [])
        if gate:
            citations = consistent_citations(
                block.get("cited_sources") or [], audit, trace.verdict, valid
            )
            if not citations:
                citations = [i for i in supporting_ids(audit, trace.verdict) if i in valid]
        else:
            citations = [i for i in (block.get("cited_sources") or []) if i in valid]
        return trace.verdict, trace.confidence, list(dict.fromkeys(citations))

    return policy


def apply_policy(records: Sequence[dict], policy: Policy, key: str = "multi") -> List[dict]:
    """Return copies of ``records`` with ``key``'s verdict replaced."""
    out: List[dict] = []
    for record in records:
        block = record.get(key)
        if not block or not block.get("verdict"):
            continue
        verdict, confidence, citations = policy(block)
        revised = dict(record)
        revised[key] = {
            **block,
            "verdict": verdict,
            "confidence": confidence,
            "cited_sources": citations,
        }
        out.append(revised)
    return out


def card_for(records: Sequence[dict], key: str) -> Scorecard:
    """Score one system over ``records``."""
    gold = [r["gold"] for r in records]
    pred = [r[key]["verdict"] for r in records]
    conf = [float(r[key].get("confidence", 0.0)) for r in records]
    secs = [float(r[key].get("seconds", 0.0)) for r in records]
    return score_system(key, gold, pred, conf, secs)


def row_for(name: str, records: Sequence[dict]) -> Dict[str, object]:
    """One ablation row: classification, calibration and citation quality."""
    card = card_for(records, "multi")
    citations = score_citations(records)
    return {
        "name": name,
        "n": card.n,
        "accuracy": 100 * card.accuracy,
        "macro_f1": 100 * card.macro_f1,
        "ece": card.ece,
        "brier": card.brier,
        "overconfidence": 100 * card.overconfidence,
        "grounded": 100 * citations.grounded_rate,
        "support": 100 * citations.support_precision,
        "n_supportable": citations.n_supportable,
    }


def latex_table(caption: str, label: str, rows: Sequence[Dict[str, object]]) -> str:
    """Render the ablation as a booktabs table."""
    header = [
        "Aggregation policy", "Acc. (\\%)", "Macro-F1 (\\%)", "ECE",
        "Brier", "Overconf. (\\%)", "Cited (\\%)", "Support (\\%)",
    ]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
        r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    str(row["name"]),
                    f"{row['accuracy']:.1f}",
                    f"{row['macro_f1']:.1f}",
                    f"{row['ece']:.3f}",
                    f"{row['brier']:.3f}",
                    f"{row['overconfidence']:+.1f}",
                    f"{row['grounded']:.1f}",
                    f"{row['support']:.1f}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL.")
    parser.add_argument("--tag", default="ablation", help="Output filename prefix.")
    parser.add_argument(
        "--corpus",
        default="",
        help="Corpus name for the caption. Defaults to a guess from --tag.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=[0.0, 1.0, 2.0],
        help="Specificity weights to sweep. beta=0 is unweighted voting.",
    )
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"No results file at {args.results}.", file=sys.stderr)
        return 1

    records = load(args.results)
    # Every claim is replayed, including the ones where retrieval
    # returned nothing. Their audit is empty, which aggregates to
    # NOT_ENOUGH_INFO at the no-evidence confidence -- exactly what the
    # run emitted for them. Dropping them would put this table on a
    # different denominator from every other table in the paper.
    usable = list(records)
    if not any((r.get("multi") or {}).get("audit") for r in usable):
        print(
            f"{args.results} carries no stored audit. Re-run evaluate.py with the\n"
            "current harness, which records the per-passage trace.",
            file=sys.stderr,
        )
        return 1

    # LaTeX captions cannot carry a raw filename: an underscore in a stem
    # such as "fever_n30_v2" is a maths-mode subscript and fails the build.
    corpus = args.corpus or {"av": "AVeriTeC", "fev": "FEVER"}.get(args.tag, args.tag)

    rows: List[Dict[str, object]] = []

    # Reference: the baseline, then the pipeline exactly as it ran.
    baseline_card = card_for(usable, "baseline")
    rows.append(
        {
            "name": "Single-LLM baseline",
            "n": baseline_card.n,
            "accuracy": 100 * baseline_card.accuracy,
            "macro_f1": 100 * baseline_card.macro_f1,
            "ece": baseline_card.ece,
            "brier": baseline_card.brier,
            "overconfidence": 100 * baseline_card.overconfidence,
            "grounded": 0.0,
            "support": 0.0,
            "n_supportable": 0,
        }
    )
    rows.append(row_for("MAFCA, as run", apply_policy(usable, policy_as_run)))

    for beta in args.betas:
        rows.append(
            row_for(
                f"\\quad + deterministic agg. ($\\beta={beta:g}$)",
                apply_policy(usable, make_deterministic(beta, gate=True)),
            )
        )

    # Isolate the citation-gating component at the default weight.
    rows.append(
        row_for(
            "\\quad + det. agg., gating off",
            apply_policy(usable, make_deterministic(1.0, gate=False)),
        )
    )

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = TAB_DIR / f"{args.tag}_ablation.tex"
    out.write_text(
        "\n".join(
            [
                f"% Generated by analysis/ablation.py from {args.results}",
                f"% n = {len(usable)} claims, replayed over stored audits.",
                "",
                latex_table(
                    f"Aggregation ablation on {corpus} by replay over fixed "
                    f"audits ($n = {len(usable)}$). Retrieval and the "
                    f"per-passage audit are identical across rows, so "
                    f"differences are attributable to the aggregation "
                    f"function alone.",
                    f"tab:{args.tag}_ablation",
                    rows,
                ),
            ]
        ),
        encoding="utf-8",
    )

    width = max(len(str(r["name"])) for r in rows) + 2
    print(f"\nAggregation ablation  ·  {args.results}  ·  n = {len(usable)}")
    print(
        f"{'policy':<{width}}{'acc':>7}{'mF1':>7}{'ECE':>8}{'Brier':>8}"
        f"{'overconf':>10}{'cited':>8}{'support':>9}"
    )
    print("-" * (width + 57))
    for row in rows:
        name = str(row["name"]).replace("\\quad ", "  ").replace("$\\beta=", "beta=").replace("$", "")
        print(
            f"{name:<{width}}{row['accuracy']:>7.1f}{row['macro_f1']:>7.1f}"
            f"{row['ece']:>8.3f}{row['brier']:>8.3f}{row['overconfidence']:>+10.1f}"
            f"{row['grounded']:>8.1f}{row['support']:>9.1f}"
        )
    print(f"\nTable -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
