"""Validate a passage cache and repair what can be repaired.

A cache written by two processes racing on the same file can lose rows,
duplicate them, or end in a half-written line. None of those announce
themselves: a truncated cache simply looks like a smaller experiment, and
a duplicated row silently double-weights one claim. Every condition
replays this file, so a fault here propagates into every number in the
paper.

    python src/analysis/validate_cache.py --cache cache/av400.jsonl --repair

Checks, in the order a fault would bite:

* **parseable** -- each line is JSON and validates against CachedClaim.
  An unparseable tail is the signature of a write interrupted mid-line.
* **unique** -- no claim id appears twice. On a duplicate the two rows
  are compared: identical ones collapse silently, differing ones are a
  genuine conflict and the richer row is kept with a warning, because
  discarding evidence is worse than keeping the fuller record.
* **complete** -- every row carries queries. A row with no queries never
  reached the search stage.
* **non-empty** -- rows whose passages are all blank are reported.
  A genuinely empty evidence set is a real outcome and is *kept*; only
  rows where retrieval never ran are suspect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.cache import CachedClaim, PassageCache  # noqa: E402


def read_lines(path: Path) -> Tuple[List[CachedClaim], List[str]]:
    """Parse every line, collecting the ones that fail rather than raising."""
    good: List[CachedClaim] = []
    bad: List[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            good.append(CachedClaim.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - any malformation is a fault.
            bad.append(f"line {number}: {type(exc).__name__}: {str(exc)[:90]}")
    return good, bad


def richer(a: CachedClaim, b: CachedClaim) -> CachedClaim:
    """Of two rows for one claim, the one carrying more evidence."""
    if len(a.passages) != len(b.passages):
        return a if len(a.passages) > len(b.passages) else b
    return a if len(a.queries) >= len(b.queries) else b


def validate(path: Path, repair: bool) -> int:
    rows, malformed = read_lines(path)

    seen: Dict[str, CachedClaim] = {}
    identical_duplicates = 0
    conflicting: List[str] = []
    for row in rows:
        existing = seen.get(row.claim_id)
        if existing is None:
            seen[row.claim_id] = row
            continue
        if existing.fingerprint() == row.fingerprint():
            identical_duplicates += 1
        else:
            conflicting.append(row.claim_id)
            seen[row.claim_id] = richer(existing, row)

    no_queries = [c for c in seen.values() if not c.queries]
    empty_evidence = [c for c in seen.values() if c.is_empty]
    blank_snippets = [
        c for c in seen.values()
        if c.passages and all(not p.snippet.strip() for p in c.passages)
    ]

    print(f"Cache: {path}")
    print(f"  lines read            {len(rows) + len(malformed)}")
    print(f"  parsed                {len(rows)}")
    print(f"  malformed             {len(malformed)}")
    for item in malformed[:5]:
        print(f"      {item}")
    print(f"  unique claim ids      {len(seen)}")
    print(f"  identical duplicates  {identical_duplicates}")
    print(f"  conflicting duplicates{len(conflicting):>6}"
          + (f"  {conflicting[:4]}" if conflicting else ""))
    print(f"  rows without queries  {len(no_queries)}")
    print(f"  rows with no passages {len(empty_evidence)}  (a real outcome; kept)")
    print(f"  rows all-blank text   {len(blank_snippets)}")

    labels: Dict[str, int] = {}
    for row in seen.values():
        labels[row.gold or "(none)"] = labels.get(row.gold or "(none)", 0) + 1
    print(f"  class counts          {dict(sorted(labels.items()))}")

    # A row that never reached search is not usable evidence for any
    # condition, and is the only category removed.
    drop = {c.claim_id for c in no_queries} | {c.claim_id for c in blank_snippets}
    survivors = {k: v for k, v in seen.items() if k not in drop}

    print(f"\n  SURVIVING ROWS        {len(survivors)} of "
          f"{len(rows) + len(malformed)} lines read")
    if drop:
        print(f"  removed               {len(drop)} (no queries, or all snippets blank)")

    if repair:
        cleaned = PassageCache(entries=survivors)
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  backup written        {backup}")
        cleaned.save(path)
        print(f"  rewrote               {path} with {len(cleaned)} rows")
    elif malformed or conflicting or drop or identical_duplicates:
        print("\n  Pass --repair to rewrite the cache with these resolved.")

    return 1 if malformed and not repair else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--repair", action="store_true",
                        help="Rewrite the cache with duplicates and unusable rows resolved.")
    args = parser.parse_args(argv)

    if not args.cache.exists():
        print(f"No cache at {args.cache}", file=sys.stderr)
        return 1
    return validate(args.cache, args.repair)


if __name__ == "__main__":
    raise SystemExit(main())
