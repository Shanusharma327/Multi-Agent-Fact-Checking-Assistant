"""Emit the revised results section's tables and macros from condition files.

The manuscript's results are rebuilt around the condition set rather than
around MAFCA-versus-B0, so the tables change shape: every row is a
condition, every classification figure carries a bootstrap interval, and
the paired comparisons are reported against B1 first because that is the
only pairing in which evidence is held fixed.

    python src/analysis/condition_tables.py --dir results/av250 \\
        --cache cache/av250.jsonl --out-dir results/tables \\
        --macros results/macros/condition_numbers.tex

Writes ``cond_classification.tex``, ``cond_differences.tex``,
``cond_calibration.tex``, ``cond_abstention.tex``, ``cond_citations.tex``
and ``cond_cost.tex``, plus one macro file holding every figure the prose
quotes. Nothing is hardcoded; a quantity the paper cites but this script
no longer emits fails the build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conditions import (  # noqa: E402
    LABELS,
    METRICS,
    ORDER,
    abstention_rate,
    accuracy_when_committing,
    comparison_family,
    common_claims,
    load_conditions,
    nei_analysis,
    cost_summary,
    rows_for,
)
from error_attribution import attribute_run  # noqa: E402
from stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    bootstrap_metric,
    paired_bootstrap_difference,
)

from fact_checker.cache import PassageCache  # noqa: E402
from fact_checker.groundedness import aggregate_span_reports  # noqa: E402
from fact_checker.retrieval_metrics import score_citations  # noqa: E402

# Short names for table rows; the long ones overflow a two-column page.
SHORT = {
    "MAJORITY": "Majority class",
    "RANDOM": "Uniform random",
    "B0": "B0: single-LLM, no retrieval",
    "B1": "B1: single-call RAG",
    "B2": "B2: MAFCA $-$ verification",
    "MAFCA": "MAFCA (agent-reported)",
    "B3": "B3: MAFCA $-$ post-conditions",
    "B4": "MAFCA (computed verdict)",
    "B6": "B6: per-passage calls",
    "ORACLE_MAFCA": "MAFCA, oracle retrieval",
    "ORACLE_B1": "B1, oracle retrieval",
}

# Macro-safe suffixes, since a macro name may hold only letters.
MACRO_KEY = {
    "MAJORITY": "Majority", "RANDOM": "Random", "B0": "Bzero", "B1": "Bone",
    "B2": "Btwo", "B3": "Bthree", "B4": "Bfour", "B6": "Bsix",
    "MAFCA": "Mafca", "ORACLE_MAFCA": "OracleMafca", "ORACLE_B1": "OracleBone",
}


def table(caption: str, label: str, spec: str, header: Sequence[str],
          body: Sequence[Sequence[str]], note: str = "") -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
        rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\begin{{tablenotes}}\small\item {note}\end{{tablenotes}}")
    lines += [r"\end{table*}", ""]
    return "\n".join(lines)


def ci(interval, places: int = 1, scale: float = 100.0) -> str:
    """A cell carrying a point estimate and its interval."""
    return (
        f"{interval.estimate * scale:.{places}f} "
        f"[{interval.low * scale:.{places}f}, {interval.high * scale:.{places}f}]"
    )


def present(order: Sequence[str], conditions: Dict) -> List[str]:
    return [name for name in order if name in conditions]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--macros", type=Path,
                        default=Path("results/macros/condition_numbers.tex"))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args(argv)

    raw = load_conditions(args.dir)
    if not raw:
        print(f"No condition files in {args.dir}", file=sys.stderr)
        return 1
    claim_ids = common_claims(raw)
    conditions = {name: rows_for(recs, claim_ids) for name, recs in raw.items()}
    names = present(ORDER + ["ORACLE_MAFCA", "ORACLE_B1"], conditions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    macros: Dict[str, str] = {}

    def put(key: str, value: str) -> None:
        macros[key] = value

    put("CondN", str(len(claim_ids)))

    # Class counts of the evaluated sample, which the reviewer required.
    first = next(iter(raw.values()))
    counts: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for claim_id in claim_ids:
        record = first[claim_id]
        counts[record.get("gold", "?")] = counts.get(record.get("gold", "?"), 0) + 1
        key = record.get("split") or "unrecorded"
        splits[key] = splits.get(key, 0) + 1
    for label, short in (("SUPPORTS", "Sup"), ("REFUTES", "Ref"),
                         ("NOT_ENOUGH_INFO", "Nei")):
        put(f"CondCount{short}", str(counts.get(label, 0)))
    put("CondTrain", str(splits.get("train", 0)))
    put("CondDev", str(splits.get("dev", 0)))

    # --- classification ---------------------------------------------------
    body = []
    for name in names:
        rows = conditions[name]
        cells = [
            ci(bootstrap_metric(rows, METRICS[m], resamples=args.resamples,
                                seed=args.seed))
            for m in ("accuracy", "macro_f1", "decidable_acc")
        ]
        body.append([SHORT.get(name, name)] + cells)
        for metric, suffix in (("accuracy", "Acc"), ("macro_f1", "FOne"),
                               ("decidable_acc", "Dec")):
            interval = bootstrap_metric(rows, METRICS[metric],
                                        resamples=args.resamples, seed=args.seed)
            key = MACRO_KEY.get(name, name)
            put(f"Cond{key}{suffix}", f"{100 * interval.estimate:.1f}")
            put(f"Cond{key}{suffix}Lo", f"{100 * interval.low:.1f}")
            put(f"Cond{key}{suffix}Hi", f"{100 * interval.high:.1f}")

    (args.out_dir / "cond_classification.tex").write_text(
        table(
            f"Classification on AVeriTeC ($n = {len(claim_ids)}$), with "
            f"{args.resamples:,}-resample bootstrap 95\\% confidence intervals. "
            "All conditions but B0 and the trivial baselines consume identical "
            "cached passages.",
            "tab:cond:class", "lccc",
            ["Condition", "Accuracy (\\%)", "Macro-F1 (\\%)", "Decidable acc. (\\%)"],
            body,
        ),
        encoding="utf-8",
    )

    # --- headline summary: the assignment's comparison, in one table -------
    if {"B0", "MAFCA", "B4"} <= set(conditions):
        head_rows = []
        for metric, label, suffix in (
            ("accuracy", "Accuracy (\\%)", "Acc"),
            ("macro_f1", "Macro-F1 (\\%)", "FOne"),
            ("decidable_acc", "Decidable accuracy (\\%)", "Dec"),
        ):
            base = METRICS[metric](conditions["B0"])
            mafca = METRICS[metric](conditions["MAFCA"])
            best = METRICS[metric](conditions["B4"])
            gap = paired_bootstrap_difference(
                list(zip(conditions["B4"], conditions["B0"])),
                lambda smp, f=METRICS[metric]: f([a for a, _ in smp]),
                lambda smp, f=METRICS[metric]: f([b for _, b in smp]),
                resamples=args.resamples,
                seed=args.seed,
            )
            head_rows.append([
                label,
                f"{100 * base:.1f}",
                f"\\textbf{{{100 * best:.1f}}}",
                f"{100 * mafca:.1f}",
                f"$\\mathbf{{{100 * (best - base):+.1f}}}$",
                f"[{100 * gap.low:.1f}, {100 * gap.high:.1f}]",
            ])
            put(f"CondGain{suffix}", f"{100 * (best - base):+.1f}")
            put(f"CondGain{suffix}Lo", f"{100 * gap.low:.1f}")
            put(f"CondGain{suffix}Hi", f"{100 * gap.high:.1f}")

        cite_base = score_citations(
            [{"multi": raw["B0"][c]} for c in claim_ids]
        ).grounded_rate
        cite_mafca = score_citations(
            [{"multi": raw["MAFCA"][c]} for c in claim_ids]
        ).grounded_rate
        cite_best = score_citations(
            [{"multi": raw["B4"][c]} for c in claim_ids]
        ).grounded_rate
        # Gain is the primary configuration on every row, including this
        # one. It previously took the agent-reported variant here and the
        # computed variant above, so one column mixed the two.
        head_rows.append([
            "Verdicts carrying a citation (\\%)",
            f"{100 * cite_base:.1f}",
            f"\\textbf{{{100 * cite_best:.1f}}}",
            f"{100 * cite_mafca:.1f}",
            f"$\\mathbf{{{100 * (cite_best - cite_base):+.1f}}}$",
            "by construction",
        ])

        (args.out_dir / "cond_headline.tex").write_text(
            table(
                f"The primary comparison on AVeriTeC ($n = {len(claim_ids)}$): "
                "the three-agent pipeline against a single LLM given the same "
                "claim and no retrieval. \\textbf{Gain is the primary "
                "configuration --- MAFCA (computed) --- over the single LLM} "
                "on every row, with its 95\\% bootstrap interval; the "
                "agent-reported column is an ablation in which the adjudicator "
                "is asked for the verdict instead of it being computed from "
                "the audit. The pipeline is ahead on every classification "
                "measure; no classification interval excludes zero at this "
                "sample size, so these are point estimates and not "
                "demonstrated differences. The citation row is a property of "
                "the architecture rather than a statistical estimate: a system "
                "that retrieves nothing cannot cite anything. The primary "
                "configuration gates citations against the audit, which is why "
                "its rate is below the ungated ablation's; "
                "Section~\\ref{sec:ablation} gives that trade-off.",
                "tab:cond:headline", "lrrrrc",
                # Stacked: six columns plus two long configuration
                # names overran the text block by 62pt.
                ["Metric", "Single LLM",
                 "\\shortstack{MAFCA\\\\(computed, primary)}",
                 "\\shortstack{MAFCA\\\\(agent-reported, ablation)}",
                 "Gain", "95\\% CI"],
                head_rows,
            ),
            encoding="utf-8",
        )

    # --- paired differences against B1 ------------------------------------
    if "B1" in conditions:
        tests = comparison_family(conditions, "B1", "accuracy",
                                  resamples=args.resamples, seed=args.seed)
        tests = [t for t in tests if t.name in names]
        body = []
        for test in tests:
            body.append([
                SHORT.get(test.name, test.name),
                ci(test.difference),
                f"{test.p_value:.3f}",
                f"{test.p_adjusted:.3f}",
                "yes" if test.inconclusive else "no",
            ])
            key = MACRO_KEY.get(test.name, test.name)
            put(f"CondDiff{key}", f"{100 * test.difference.estimate:+.1f}")
            put(f"CondDiff{key}Lo", f"{100 * test.difference.low:.1f}")
            put(f"CondDiff{key}Hi", f"{100 * test.difference.high:.1f}")
            put(f"CondDiff{key}P", f"{test.p_adjusted:.3f}")
        (args.out_dir / "cond_differences.tex").write_text(
            table(
                "Paired accuracy differences against B1, the single-call RAG "
                "baseline, by exact McNemar with Holm correction across the "
                "family. A confidence interval spanning zero means no "
                "difference was demonstrated.",
                "tab:cond:diff", "lcccc",
                ["Condition $-$ B1", "Difference (pp)", "$p$", "$p$ (Holm)",
                 "CI spans 0"],
                body,
            ),
            encoding="utf-8",
        )

    # --- calibration -------------------------------------------------------
    body = []
    for name in names:
        rows = conditions[name]
        cells = [
            ci(bootstrap_metric(rows, METRICS[m], resamples=args.resamples,
                                seed=args.seed), places=3, scale=1.0)
            for m in ("ece10", "ece15", "brier")
        ]
        body.append([SHORT.get(name, name)] + cells)
        key = MACRO_KEY.get(name, name)
        for metric, suffix in (("ece10", "Ece"), ("brier", "Brier")):
            value = METRICS[metric](rows)
            put(f"Cond{key}{suffix}", f"{value:.3f}")
    (args.out_dir / "cond_calibration.tex").write_text(
        table(
            "Calibration. Lower is better throughout. The trivial baselines "
            "score well by reporting the uninformative prior and being right "
            "about as often, which is why calibration is never read as a "
            "quality measure on its own.",
            "tab:cond:calib", "lccc",
            ["Condition", "ECE (10 bins)", "ECE (15 bins)", "Brier"],
            body,
        ),
        encoding="utf-8",
    )

    # --- abstention --------------------------------------------------------
    body = []
    for name in names:
        rows = conditions[name]
        nei = nei_analysis(raw[name], claim_ids)
        def pct(value: float) -> str:
            return "---" if value != value else f"{100 * value:.1f}"
        body.append([
            SHORT.get(name, name),
            f"{100 * abstention_rate(rows):.1f}",
            f"{100 * accuracy_when_committing(rows):.1f}",
            pct(nei["nei_recall_all"]),
            pct(nei["accuracy_with_evidence"]),
        ])
        key = MACRO_KEY.get(name, name)
        put(f"Cond{key}Abstain", f"{100 * abstention_rate(rows):.1f}")
        put(f"Cond{key}AccCommit", f"{100 * accuracy_when_committing(rows):.1f}")
    (args.out_dir / "cond_abstention.tex").write_text(
        table(
            "Abstention. A system can raise overall accuracy by declining more "
            "often, so the share of \\textsc{not\\_enough\\_info} verdicts is "
            "reported beside the accuracy of the verdicts actually committed to.",
            "tab:cond:abstain", "lcccc",
            ["Condition", "Abstains (\\%)", "Acc.\\ when committing (\\%)",
             "NEI recall (\\%)", "Acc.\\ with evidence (\\%)"],
            body,
        ),
        encoding="utf-8",
    )

    # --- citations and cost ------------------------------------------------
    body = []
    for name in names:
        records = [raw[name][c] for c in claim_ids]
        citations = score_citations([{"multi": r} for r in records])
        cost = cost_summary(raw[name], claim_ids)
        body.append([
            SHORT.get(name, name),
            f"{100 * citations.grounded_rate:.1f}",
            f"{100 * citations.validity:.1f}" if citations.n_citations else "---",
            f"{100 * citations.support_precision:.1f}" if citations.n_supportable else "---",
            f"{cost['mean_calls']:.0f}",
        ])
        key = MACRO_KEY.get(name, name)
        put(f"Cond{key}Cited", f"{100 * citations.grounded_rate:.1f}")
        put(f"Cond{key}Support", f"{100 * citations.support_precision:.1f}")
        put(f"Cond{key}Calls", f"{cost['mean_calls']:.0f}")
    (args.out_dir / "cond_citations.tex").write_text(
        table(
            "Citation behaviour and inference cost. Validity is guaranteed by "
            "construction wherever the citation post-condition is active and is "
            "therefore not an empirical result; \\emph{structural} support --- "
            "the verification stage having labelled the cited passage in the "
            "verdict's direction --- is not guaranteed. Neither is a semantic "
            "judgement that the passage entails the claim. Calls exclude "
            "retrieval, which is cached.",
            "tab:cond:cite", "lcccc",
            ["Condition", "Verdicts cited (\\%)", "Validity (\\%)",
             "Structural support (\\%)", "LLM calls"],
            body,
        ),
        encoding="utf-8",
    )

    # --- error attribution and the span diagnostic -------------------------
    if "MAFCA" in raw:
        attribution = attribute_run([raw["MAFCA"][c] for c in claim_ids])
        for stage in ("retrieval", "verification", "aggregation"):
            put(f"CondErr{stage.capitalize()}",
                f"{100 * attribution['shares'][stage]:.1f}")
        put("CondErrWrong", str(attribution["n_wrong"]))

        cache = PassageCache.load(args.cache) if args.cache else None
        reports = []
        if cache:
            from fact_checker.groundedness import audit_span_report
            from fact_checker.schemas import Alignment, AuditReport
            for claim_id in claim_ids:
                record = raw["MAFCA"][claim_id]
                entry = cache.get(claim_id)
                if entry is None or not record.get("audit"):
                    continue
                audit = AuditReport(alignments=[
                    Alignment(source_id=a["source_id"], verdict=a["verdict"],
                              justification=a.get("justification", ""))
                    for a in record["audit"] if a.get("source_id")
                ])
                reports.append(audit_span_report(audit, entry.to_scout_report()).model_dump())
        if reports:
            spans = aggregate_span_reports(reports)
            put("CondQuoteRate", f"{100 * spans['quote_rate']:.1f}")
            put("CondQuoteCount", str(spans["n_quoting"]))
            put("CondAlignments", str(spans["n_alignments"]))
            put("CondSpanCheckable", str(spans["n_checkable"]))
            put("CondSpanViolations", str(spans["n_violating"]))
            put("CondSpanRate", f"{100 * spans['violation_rate']:.1f}")
            put("CondSpanNegated", str(spans["n_negated_spans"]))

    args.macros.parent.mkdir(parents=True, exist_ok=True)
    args.macros.write_text(
        "\n".join(
            [f"% Generated by analysis/condition_tables.py from {args.dir}",
             f"% n = {len(claim_ids)} claims. Do not edit by hand.", ""]
            + [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
            + [""]
        ),
        encoding="utf-8",
    )

    print(f"n = {len(claim_ids)} claims, conditions: {', '.join(names)}")
    print(f"  tables -> {args.out_dir}")
    print(f"  macros -> {args.macros}  ({len(macros)} definitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
