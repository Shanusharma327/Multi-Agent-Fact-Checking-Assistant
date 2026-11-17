"""Reshape the per-condition run into the paired record the reporters expect.

The 250-claim protocol writes one file per condition
(``results/av250/B0.jsonl``, ``MAFCA.jsonl``, ...), one flat record per
claim. The reporting chain that produces ``results/macros/numbers.tex`` --
``results_report.py``, ``paper_numbers.py``, ``ablation.py`` and the
figure scripts -- predates it and reads the shape ``evaluate.py`` emits:
one record per claim carrying a ``baseline`` block and a ``multi`` block
side by side.

Without this join the two halves of the Results section describe
different runs: the ``\\Av`` macros come from the earlier 60-claim pilot
and the ``\\Cond`` macros from the 90-claim run, and a reader has no way
to tell. This script joins the condition files on claim id so that one
run feeds both.

Nothing here computes a result. Fields are copied, and the only field
reconstructed is the passage text, which the condition files reference
by identifier and the passage cache stores in full; the join asserts
that the identifiers agree rather than trusting them to.

    python src/analysis/join_conditions.py \\
        --conditions results/av250 --cache cache/av250.jsonl \\
        --out results/averitec_n90_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# A claim's three agent stages are not timed separately by the condition
# runner, so the per-stage breakdown the pilot recorded is unavailable
# and is omitted rather than apportioned from the total.
BASELINE_FIELDS = ("verdict", "confidence", "reasoning", "seconds")

MULTI_FIELDS = (
    "verdict",
    "confidence",
    "reasoning",
    "seconds",
    "cited_sources",
    "evidence_ids",
    "n_evidence",
    "queries",
    "retrieved_urls",
    "audit",
    "audit_tally",
    "aggregation_trace",
    "span_check",
    "llm_calls",
    "short_circuit",
)

CLAIM_FIELDS = (
    "id",
    "claim",
    "gold",
    "dataset",
    "gold_urls",
    "fact_check_url",
    "split",
    "errors",
)


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(records: List[dict], key: str = "id") -> Dict[str, dict]:
    return {str(r[key]): r for r in records}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=Path("results/av250"))
    parser.add_argument("--cache", type=Path, default=Path("cache/av250.jsonl"))
    parser.add_argument(
        "--baseline",
        default="B0",
        help="Condition supplying the single-LLM column.",
    )
    parser.add_argument(
        "--multi",
        default="MAFCA",
        help="Condition supplying the three-agent column.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/averitec_n90_v2.jsonl")
    )
    args = parser.parse_args(argv)

    base_path = args.conditions / f"{args.baseline}.jsonl"
    multi_path = args.conditions / f"{args.multi}.jsonl"
    for path in (base_path, multi_path, args.cache):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    baseline = by_id(read_jsonl(base_path))
    multi = by_id(read_jsonl(multi_path))
    cache = by_id(read_jsonl(args.cache), key="claim_id")

    shared = sorted(set(baseline) & set(multi))
    if not shared:
        print("no claims in common between the two conditions", file=sys.stderr)
        return 1
    dropped = (set(baseline) | set(multi)) - set(shared)
    if dropped:
        print(f"note: {len(dropped)} claim(s) present in only one condition, skipped")

    out: List[dict] = []
    for claim_id in shared:
        b, m = baseline[claim_id], multi[claim_id]
        if b["gold"] != m["gold"]:
            print(f"gold label disagrees for {claim_id}", file=sys.stderr)
            return 1

        record = {field: m.get(field) for field in CLAIM_FIELDS}
        # The pilot schema carries this field; nothing downstream reads it
        # for AVeriTeC, and inventing page-level gold would be worse than
        # leaving it empty.
        record["gold_pages"] = []
        record["retrieval"] = "live"
        record["aggregation"] = "prompt"
        record["baseline"] = {f: b.get(f) for f in BASELINE_FIELDS}
        record["multi"] = {f: m.get(f) for f in MULTI_FIELDS}

        entry = cache.get(claim_id)
        passages = list(entry.get("passages", [])) if entry else []
        # The evidence the auditor saw is identified by source_id in the
        # condition file and stored in full in the cache. Agreement is
        # asserted, not assumed: a mismatch means the two artefacts came
        # from different runs and every retrieval figure would be wrong.
        ids_in_run = list(m.get("evidence_ids") or [])
        ids_in_cache = [p.get("source_id") for p in passages]
        if ids_in_run and ids_in_cache[: len(ids_in_run)] != ids_in_run:
            print(
                f"evidence identifiers disagree for {claim_id}:\n"
                f"  run:   {ids_in_run}\n"
                f"  cache: {ids_in_cache}",
                file=sys.stderr,
            )
            return 1
        record["multi"]["evidence"] = passages[: len(ids_in_run)]
        record["multi"]["warnings"] = (
            list(entry.get("retrieval_warnings", [])) if entry else []
        )
        out.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in out:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    labels: Dict[str, int] = {}
    for record in out:
        labels[record["gold"]] = labels.get(record["gold"], 0) + 1
    print(f"wrote {len(out)} paired records to {args.out}")
    print(f"  baseline = {args.baseline}, multi = {args.multi}")
    print("  labels: " + ", ".join(f"{k} {v}" for k, v in sorted(labels.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
