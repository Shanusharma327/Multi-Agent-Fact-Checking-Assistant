"""Download the benchmark splits this project evaluates on.

The raw corpora are not committed: AVeriTeC is ~12 MB and FEVER ~4 MB,
and both are redistributed by their own authors under their own terms.
This script fetches them into ``data/`` using the same loaders the
evaluation uses, so what you get is what the run read.

    python data/download_datasets.py            # AVeriTeC dev + train
    python data/download_datasets.py --fever    # also FEVER dev

Note that downloading the corpora is only needed to draw a *new* sample
of claims. Reproducing the reported numbers does not need them: the
retrieved passages are committed in ``cache/av250.jsonl`` and the
per-claim verdicts in ``results/``, and
``python src/analysis/regenerate.py`` rebuilds every table and figure
from those alone, with no network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fact_checker.datasets.averitec import load_averitec  # noqa: E402
from fact_checker.datasets.fever import load_fever  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fever", action="store_true",
        help="Also fetch FEVER. Not needed: the report withdrew the FEVER pilot.",
    )
    parser.add_argument(
        "--splits", nargs="*", default=["dev", "train"],
        help="AVeriTeC splits to fetch (default: dev train).",
    )
    args = parser.parse_args(argv)

    averitec_dir = ROOT / "data" / "averitec"
    for split in args.splits:
        print(f"AVeriTeC {split} -> {averitec_dir}")
        claims, report = load_averitec(split, averitec_dir, download=True)
        print(f"  {len(claims)} claims   label counts: {report}")

    if args.fever:
        fever_dir = ROOT / "data" / "fever"
        print(f"FEVER dev -> {fever_dir}")
        claims = load_fever("dev", fever_dir, download=True)
        print(f"  {len(claims)} claims")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
