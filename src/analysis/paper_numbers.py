"""Emit every number the paper quotes as a LaTeX macro.

Hand-transcribing results into prose is how a paper comes to disagree with
its own code: a run is repeated, one table is updated, and a sentence three
pages away still quotes the old figure. This script computes each quantity
from the stored per-claim output and writes it as a ``\\newcommand``, so the
paper cites a macro and the macro cites the run.

    python src/analysis/paper_numbers.py \\
        --averitec results/averitec_n60_v2.jsonl \\
        --fever    results/fever_n30_v2.jsonl \\
        --out      results/macros/numbers.tex

Any macro the paper uses but this script does not define is an undefined
control sequence at compile time -- a loud failure rather than a silent
stale number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation import apply_policy, card_for, make_deterministic, policy_as_run  # noqa: E402

from fact_checker.cache import PassageCache  # noqa: E402
from fact_checker.groundedness import (  # noqa: E402
    aggregate_span_reports,
    audit_span_report,
)
from fact_checker.schemas import Alignment, AuditReport  # noqa: E402
from fact_checker.metrics import Scorecard, score_subset  # noqa: E402
from fact_checker.retrieval_metrics import (  # noqa: E402
    leakage_exposure,
    score_citations,
    score_retrieval,
)

DECIDABLE = ("SUPPORTS", "REFUTES")


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Macros:
    """Accumulates ``\\newcommand`` definitions, rejecting redefinitions."""

    def __init__(self) -> None:
        self._items: Dict[str, str] = {}

    def add(self, name: str, value: str) -> None:
        if name in self._items and self._items[name] != value:
            raise ValueError(f"macro {name!r} defined twice with different values")
        self._items[name] = value

    def num(self, name: str, value: float, places: int = 1) -> None:
        self.add(name, f"{value:.{places}f}")

    def signed(self, name: str, value: float, places: int = 1) -> None:
        self.add(name, f"{value:+.{places}f}")

    def render(self, header: Sequence[str]) -> str:
        lines = [f"% {line}" for line in header]
        lines.append("")
        for name in sorted(self._items):
            lines.append(rf"\newcommand{{\{name}}}{{{self._items[name]}}}")
        lines.append("")
        return "\n".join(lines)


def decidable_accuracy(records: Sequence[dict], key: str = "multi") -> float:
    """Accuracy restricted to claims whose gold label is decidable.

    Reported apart from the headline because NOT_ENOUGH_INFO is scored
    against a corpus, not against the world, so a system that over-predicts
    it can look good overall while failing on every claim that had an answer.
    """
    gold = [r["gold"] for r in records]
    pred = [r[key]["verdict"] for r in records]
    conf = [float(r[key].get("confidence", 0.0)) for r in records]
    return 100 * score_subset(key, gold, pred, conf, DECIDABLE).accuracy


def _span_reports(records: Sequence[dict], cache: "PassageCache | None") -> List[dict]:
    """Span-check reports for a run, recomputed where the cache allows it.

    Falls back to the value stored at run time only when no cache is
    available, and that fallback is the stale case the recomputation
    exists to avoid -- so it is used, not silently preferred.
    """
    reports: List[dict] = []
    for record in records:
        block = record.get("multi") or record
        audit_rows = block.get("audit") or []
        entry = cache.get(str(record.get("id", ""))) if cache else None

        if entry is not None and audit_rows:
            audit = AuditReport(
                alignments=[
                    Alignment(
                        source_id=str(a.get("source_id", "")),
                        verdict=a.get("verdict", "NOT_ENOUGH_INFO"),
                        justification=str(a.get("justification", "")),
                    )
                    for a in audit_rows
                    if a.get("source_id")
                ]
            )
            reports.append(audit_span_report(audit, entry.to_scout_report()).model_dump())
        elif block.get("span_check"):
            reports.append(block["span_check"])
    return reports


def card_macros(macros: Macros, prefix: str, card: Scorecard) -> None:
    """Classification and calibration figures for one system."""
    macros.add(f"{prefix}N", str(card.n))
    macros.num(f"{prefix}Acc", 100 * card.accuracy)
    macros.num(f"{prefix}MacroFOne", 100 * card.macro_f1)
    macros.num(f"{prefix}Err", 100 * (1 - card.accuracy))
    macros.num(f"{prefix}Conf", 100 * card.mean_confidence)
    macros.signed(f"{prefix}Overconf", 100 * card.overconfidence)
    macros.num(f"{prefix}Ece", card.ece, 3)
    macros.num(f"{prefix}Brier", card.brier, 3)
    macros.num(f"{prefix}Latency", card.mean_seconds)
    macros.num(
        f"{prefix}MacroPrec",
        100 * statistics.mean(s.precision for s in card.per_label),
    )
    macros.num(
        f"{prefix}MacroRec", 100 * statistics.mean(s.recall for s in card.per_label)
    )
    short = {"SUPPORTS": "Sup", "REFUTES": "Ref", "NOT_ENOUGH_INFO": "Nei"}
    for score in card.per_label:
        tag = short[score.label]
        macros.num(f"{prefix}Prec{tag}", 100 * score.precision)
        macros.num(f"{prefix}Rec{tag}", 100 * score.recall)
        macros.num(f"{prefix}FOne{tag}", 100 * score.f1)
        macros.add(f"{prefix}Support{tag}", str(score.support))
        # support is how often the corpus used this label; this is how
        # often the system did. Precision over two predictions and
        # precision over forty are different evidence.
        issued = sum(row.get(score.label, 0) for row in card.confusion.values())
        macros.add(f"{prefix}Pred{tag}", str(issued))


def dataset_macros(
    macros: Macros,
    prefix: str,
    records: Sequence[dict],
    cache: "PassageCache | None" = None,
) -> None:
    """Every quantity the paper reports for one corpus."""
    paired = [
        r
        for r in records
        if (r.get("multi") or {}).get("verdict") and (r.get("baseline") or {}).get("verdict")
    ]
    # A condition-format file carries one system per file, with its fields
    # at the top level. Everything below that needs a (multi, baseline)
    # pair is skipped for such a file; the span diagnostic is not.
    usable = paired or [r for r in records if r.get("verdict")]
    single_system = not paired
    macros.add(f"{prefix}Claims", str(len(usable)))

    if not single_system:
        card_macros(macros, f"{prefix}Base", card_for(usable, "baseline"))
        card_macros(macros, f"{prefix}Multi", card_for(usable, "multi"))

    # Deterministic aggregation, replayed over the same audits.
    #
    # Every claim is replayed, including the ones where retrieval
    # returned nothing and the audit is therefore empty. Those claims
    # still produced a verdict, and the deterministic policy reproduces
    # it: an empty audit aggregates to NOT_ENOUGH_INFO at the
    # no-evidence confidence. Dropping them would leave the replay rows
    # on a different denominator from the baseline row beside them, so
    # the comparison would not be paired.
    traced = [] if single_system else list(usable)
    if traced:
        for beta, tag in ((0.0, "Zero"), (1.0, ""), (2.0, "Two")):
            rows = apply_policy(traced, make_deterministic(beta, gate=True))
            card_macros(macros, f"{prefix}Det{tag}", card_for(rows, "multi"))
            macros.num(f"{prefix}Det{tag}AccDecidable", decidable_accuracy(rows))
            citations = score_citations(rows)
            macros.num(f"{prefix}Det{tag}Grounded", 100 * citations.grounded_rate)
            # The count behind the rate. Section 4.7 quotes "N of 90",
            # and which N depends on the configuration.
            macros.add(f"{prefix}Det{tag}Cited", str(citations.n_with_citations))
            macros.num(f"{prefix}Det{tag}Support", 100 * citations.support_precision)
            macros.add(f"{prefix}Det{tag}Supportable", str(citations.n_supportable))

        ungated = apply_policy(traced, make_deterministic(1.0, gate=False))
        card_macros(macros, f"{prefix}DetUngated", card_for(ungated, "multi"))
        macros.num(
            f"{prefix}DetUngatedGrounded",
            100 * score_citations(ungated).grounded_rate,
        )
        macros.add(
            f"{prefix}DetUngatedCited",
            str(score_citations(ungated).n_with_citations),
        )
        macros.num(
            f"{prefix}DetUngatedSupport",
            100 * score_citations(ungated).support_precision,
        )
        macros.add(f"{prefix}Traced", str(len(traced)))

        # The agent-reported condition, scored identically for comparison.
        as_run = apply_policy(traced, policy_as_run)
        card_macros(macros, f"{prefix}Prompt", card_for(as_run, "multi"))
        macros.num(f"{prefix}PromptAccDecidable", decidable_accuracy(as_run))
        prompt_cites = score_citations(as_run)
        macros.num(f"{prefix}PromptGrounded", 100 * prompt_cites.grounded_rate)
        macros.add(f"{prefix}PromptCited", str(prompt_cites.n_with_citations))
        macros.num(f"{prefix}PromptSupport", 100 * prompt_cites.support_precision)
        macros.add(f"{prefix}PromptSupportable", str(prompt_cites.n_supportable))

    # Equation (13) diagnostic. Reported as coverage first and rate
    # second, because the rate is meaningless without knowing how small
    # a share of judgements it was computed over.
    span_reports = _span_reports(usable, cache)
    if span_reports:
        spans = aggregate_span_reports(span_reports)
        macros.add(f"{prefix}AlignmentCount", str(spans["n_alignments"]))
        macros.add(f"{prefix}QuoteCount", str(spans["n_quoting"]))
        macros.num(f"{prefix}QuoteRate", 100 * spans["quote_rate"])
        macros.add(f"{prefix}SpanCheckable", str(spans["n_checkable"]))
        macros.add(f"{prefix}SpanViolations", str(spans["n_violating"]))
        macros.num(f"{prefix}SpanViolationRate", 100 * spans["violation_rate"])
        macros.add(f"{prefix}SpanNegated", str(spans["n_negated_spans"]))

    # Decidable subset, reported apart from NOT_ENOUGH_INFO.
    pairs = () if single_system else (("baseline", "Base"), ("multi", "Multi"))
    for key, tag in pairs:
        gold = [r["gold"] for r in usable]
        pred = [r[key]["verdict"] for r in usable]
        conf = [float(r[key].get("confidence", 0.0)) for r in usable]
        macros.num(
            f"{prefix}{tag}AccDecidable",
            100 * score_subset(key, gold, pred, conf, DECIDABLE).accuracy,
        )

    if single_system:
        # A condition-format file carries one system; the paired
        # metrics below have no second arm to compare against.
        return

    # Citation counts on the run as it stands.
    citations = score_citations(usable)
    macros.add(f"{prefix}MultiCited", str(citations.n_with_citations))
    macros.num(f"{prefix}MultiGrounded", 100 * citations.grounded_rate)
    macros.num(f"{prefix}MultiValidity", 100 * citations.validity)

    # Retrieval, where the corpus annotates gold evidence URLs.
    scorable = [
        r
        for r in usable
        if (r.get("multi") or {}).get("retrieved_urls") is not None and r.get("gold_urls")
    ]
    if scorable:
        retrieved = [r["multi"]["retrieved_urls"] for r in scorable]
        gold = [r["gold_urls"] for r in scorable]
        for domain, tag in ((False, "Exact"), (True, "Domain")):
            score = score_retrieval(retrieved, gold, domain=domain)
            macros.add(f"{prefix}Ret{tag}N", str(score.n_claims))
            macros.num(f"{prefix}Ret{tag}Recall", 100 * score.recall_at_k)
            macros.num(f"{prefix}Ret{tag}Precision", 100 * score.precision_at_k)
            macros.num(f"{prefix}Ret{tag}Hit", 100 * score.hit_rate)
            macros.num(f"{prefix}Ret{tag}MeanRetrieved", score.mean_retrieved, 2)
            macros.num(f"{prefix}Ret{tag}MeanGold", score.mean_gold, 2)

        exposure = leakage_exposure(usable)
        macros.add(f"{prefix}LeakFiltered", str(int(exposure["n_filtered_claims"])))
        macros.add(f"{prefix}LeakExposed", str(int(exposure["n_exposed"])))
        macros.num(f"{prefix}LeakRate", 100 * exposure["rate"])
        macros.add(f"{prefix}LeakDomains", str(int(exposure["n_domains"])))

    # Evidence and query diagnostics.
    evidence_counts = [int(r["multi"].get("n_evidence", 0)) for r in usable]
    if evidence_counts:
        macros.num(f"{prefix}MeanEvidence", statistics.mean(evidence_counts), 2)
        macros.add(
            f"{prefix}ZeroEvidence", str(sum(1 for n in evidence_counts if n == 0))
        )
    queries = [len(r["multi"].get("queries") or []) for r in usable]
    if queries:
        macros.add(f"{prefix}TwoQuery", str(sum(1 for q in queries if q >= 2)))

    # Latency ratio, the subject of RQ3.
    base_latency = statistics.mean(float(r["baseline"].get("seconds", 0.0)) for r in usable)
    multi_latency = statistics.mean(float(r["multi"].get("seconds", 0.0)) for r in usable)
    macros.num(
        f"{prefix}LatencyRatio", multi_latency / base_latency if base_latency else 0.0, 1
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--averitec", type=Path, help="AVeriTeC results JSONL.")
    parser.add_argument("--fever", type=Path, help="FEVER results JSONL.")
    parser.add_argument("--oracle", type=Path, help="Oracle-retrieval results JSONL.")
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Passage cache, so the span diagnostic is recomputed from the "
             "trace rather than read from a possibly stale stored value.")
    parser.add_argument("--out", type=Path, default=Path("results/macros/numbers.tex"))
    args = parser.parse_args(argv)

    macros = Macros()
    header = ["Generated by analysis/paper_numbers.py. Do not edit by hand."]

    sources = [("Av", args.averitec), ("Fev", args.fever), ("Or", args.oracle)]
    for prefix, path in sources:
        if path is None:
            continue
        if not path.exists():
            print(f"skipping {prefix}: no file at {path}", file=sys.stderr)
            continue
        records = load(path)
        cache = PassageCache.load(args.cache) if args.cache else None
        dataset_macros(macros, prefix, records, cache)
        header.append(f"{prefix}: {path} ({len(records)} records)")

    if len(header) == 1:
        print("Nothing to do: pass at least one results file.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(macros.render(header), encoding="utf-8")
    print(f"wrote {len(macros._items)} macros -> {args.out}")
    for line in header[1:]:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
