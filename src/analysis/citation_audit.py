"""Export a blind sample of citations for human annotation.

Citation *support* as the pipeline reports it is structural: a citation
counts as supported when the verification stage labelled that passage in
the verdict's direction. That is checkable in code and it is not a
semantic judgement -- it inherits whatever the verification agent got
wrong. The only way to know how often a cited passage genuinely bears on
the verdict is for a person to read them.

This writes that sample as a CSV. Three properties make it a *blind*
sample rather than a presentation of the system's own opinion:

* The system's structural verdict on each row is **not** exported. An
  annotator who can see that the pipeline already called a citation
  supported is no longer an independent check of it.
* Rows are drawn by stratified random sampling across verdict classes
  and shuffled under a fixed seed, so the order carries no signal and
  the draw is reproducible.
* The passage is given in full, not truncated to the span the model
  quoted, so the annotator can see context the model may have ignored.

    python src/analysis/citation_audit.py \\
        --results results/av400/MAFCA.jsonl \\
        --cache cache/av400.jsonl \\
        --out analysis/citation_audit_sample.csv

Fill the ``human_supports`` column with ``yes`` / ``no`` / ``unclear``,
then run ``--score`` on the completed file to get agreement against the
structural check.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.cache import PassageCache  # noqa: E402

ANSWERS = ("yes", "no", "unclear")

COLUMNS = [
    "row_id",
    "claim_id",
    "claim",
    "verdict",
    "cited_source_id",
    "cited_passage",
    "model_justification",
    "human_supports",
    "human_notes",
]

INSTRUCTIONS = [
    "# Citation support annotation",
    "#",
    "# For each row: does the CITED PASSAGE provide evidence for the VERDICT",
    "# on the CLAIM? Judge the passage against the claim only. Ignore whether",
    "# the model's justification sounds convincing, and ignore anything you",
    "# know about the claim that the passage does not say.",
    "#",
    "#   yes     - the passage supports the verdict's direction",
    "#   no      - the passage does not, or points the other way",
    "#   unclear - genuinely cannot tell from this passage alone",
    "#",
    "# Leave human_notes blank unless something is worth recording.",
    "# Do not reorder or delete rows; blank answers are reported as skipped.",
]


def load_rows(results: Path, cache: PassageCache) -> List[dict]:
    """Every (verdict, citation) pair in a results file, with its passage."""
    records = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: List[dict] = []
    for record in records:
        if not record.get("verdict") or not record.get("cited_sources"):
            continue
        entry = cache.get(str(record.get("id", "")))
        if entry is None:
            continue
        passages = {p.source_id: p.snippet for p in entry.passages}
        justifications = {
            a["source_id"]: a.get("justification", "")
            for a in (record.get("audit") or [])
        }
        audit_labels = {
            a["source_id"]: a.get("verdict", "") for a in (record.get("audit") or [])
        }
        for source_id in record["cited_sources"]:
            if source_id not in passages:
                continue
            out.append(
                {
                    "claim_id": record["id"],
                    "claim": record.get("claim", ""),
                    "verdict": record["verdict"],
                    "cited_source_id": source_id,
                    "cited_passage": passages[source_id],
                    "model_justification": justifications.get(source_id, ""),
                    # Withheld from the CSV; kept here for scoring later.
                    "_structural_supported": audit_labels.get(source_id) == record["verdict"],
                }
            )
    return out


def draw(rows: Sequence[dict], size: int, seed: int) -> List[dict]:
    """Stratify by verdict, then shuffle, so the order carries no signal."""
    rng = random.Random(seed)
    by_verdict: Dict[str, List[dict]] = {}
    for row in rows:
        by_verdict.setdefault(row["verdict"], []).append(row)

    per_class = max(1, size // max(1, len(by_verdict)))
    selected: List[dict] = []
    for verdict in sorted(by_verdict):
        pool = by_verdict[verdict]
        rng.shuffle(pool)
        selected.extend(pool[:per_class])

    # Top up from whatever remains if a class was short.
    if len(selected) < size:
        chosen = {id(r) for r in selected}
        remainder = [r for r in rows if id(r) not in chosen]
        rng.shuffle(remainder)
        selected.extend(remainder[: size - len(selected)])

    rng.shuffle(selected)
    return selected[:size]


def write_csv(rows: Sequence[dict], out: Path, key_out: Path) -> None:
    """Write the blind CSV, and the withheld answer key beside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        for line in INSTRUCTIONS:
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "row_id": index,
                    "claim_id": row["claim_id"],
                    "claim": row["claim"],
                    "verdict": row["verdict"],
                    "cited_source_id": row["cited_source_id"],
                    "cited_passage": row["cited_passage"],
                    "model_justification": row["model_justification"],
                    "human_supports": "",
                    "human_notes": "",
                }
            )

    # The structural verdict is kept out of the annotator's file entirely.
    with key_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "claim_id", "cited_source_id", "structural_supported"])
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                [index, row["claim_id"], row["cited_source_id"],
                 "yes" if row["_structural_supported"] else "no"]
            )


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over a shared label set."""
    if not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum(
        (sum(1 for x in a if x == label) / n) * (sum(1 for y in b if y == label) / n)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score(filled: Path, key: Path) -> int:
    """Compare completed annotations against the structural check."""
    answers: Dict[int, str] = {}
    with filled.open("r", encoding="utf-8", newline="") as handle:
        lines = [l for l in handle if not l.startswith("#")]
    for row in csv.DictReader(lines):
        value = (row.get("human_supports") or "").strip().lower()
        if value in ANSWERS:
            answers[int(row["row_id"])] = value

    structural: Dict[int, str] = {}
    with key.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            structural[int(row["row_id"])] = row["structural_supported"]

    shared = sorted(set(answers) & set(structural))
    if not shared:
        print("No completed annotations found. Fill human_supports first.",
              file=sys.stderr)
        return 1

    # 'unclear' is excluded from agreement: it is a refusal to judge, not a
    # third opinion the structural check could ever express.
    decidable = [r for r in shared if answers[r] in ("yes", "no")]
    human = [answers[r] for r in decidable]
    machine = [structural[r] for r in decidable]

    agree = sum(1 for h, m in zip(human, machine) if h == m)
    human_yes = sum(1 for h in human if h == "yes")

    print(f"annotated rows        {len(answers)} of {len(structural)}")
    print(f"  unclear             {sum(1 for r in shared if answers[r] == 'unclear')}")
    print(f"  decidable           {len(decidable)}")
    print(f"raw agreement         {100 * agree / len(decidable):.1f}%"
          if decidable else "raw agreement         n/a")
    print(f"Cohen's kappa         {cohens_kappa(human, machine):.3f}")
    print(f"human says supported  {100 * human_yes / len(decidable):.1f}%"
          if decidable else "")
    print(
        f"structural says so    "
        f"{100 * sum(1 for m in machine if m == 'yes') / len(decidable):.1f}%"
        if decidable else ""
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="MAFCA results JSONL.")
    parser.add_argument("--cache", type=Path, help="Passage cache JSONL.")
    parser.add_argument("--out", type=Path,
                        default=Path("results/citation_audit_sample.csv"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--score", type=Path, default=None,
                        help="Score a completed CSV instead of exporting.")
    args = parser.parse_args(argv)

    key_path = args.out.with_name(args.out.stem + "_key.csv")

    if args.score:
        return score(args.score, key_path)

    if not args.results or not args.cache:
        print("--results and --cache are required for export.", file=sys.stderr)
        return 1

    cache = PassageCache.load(args.cache)
    rows = load_rows(args.results, cache)
    if not rows:
        print(f"No citations found in {args.results}", file=sys.stderr)
        return 1

    sample = draw(rows, args.size, args.seed)
    write_csv(sample, args.out, key_path)

    by_verdict: Dict[str, int] = {}
    for row in sample:
        by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1

    print(f"{len(rows)} citations available; sampled {len(sample)}")
    print(f"  by verdict    {dict(sorted(by_verdict.items()))}")
    print(f"  annotate      {args.out}")
    print(f"  answer key    {key_path}  (withheld from the CSV above)")
    print(f"\nWhen done:  python src/analysis/citation_audit.py --score {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
