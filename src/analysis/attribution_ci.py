"""Bootstrap confidence intervals on the error-attribution shares.

    python src/analysis/attribution_ci.py

The attribution splits each wrong verdict across retrieval,
verification and aggregation. Those shares are the paper's most
specific claim about where effort should go next, and they were
reported as bare percentages over 47 live errors and 37 oracle ones --
small denominators on which a few claims moving between stages shifts a
share by several points.

Resampling is over *claims*, not over errors, because the claim is the
sampling unit: a resample that drops a claim drops its error too, and
the number of errors therefore varies between resamples exactly as it
would between studies. Each resample re-runs the attribution cascade
and recomputes the share, so the interval carries both sources of
variation.

B1 is reported only if its records support attribution. They do not:
the cascade reads the per-passage audit, and a single-call condition
emits no audit. That is itself the point the paper makes about
monolithic systems, so it is stated rather than worked around.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/analysis/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from ablation import apply_policy, make_deterministic  # noqa: E402
from error_attribution import STAGES, attribute, attribute_run  # noqa: E402

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261018

# The macro name each stage contributes to, matching the existing
# \CondErrLive... / \CondErrOracle... family in condition_numbers.tex.
STAGE_TAG = {
    "retrieval": "Retrieval",
    "verification": "Verification",
    "aggregation": "Aggregation",
    "unattributed": "Unattributed",
}


def load(path: Path, *, replay: bool = False) -> List[dict]:
    """Records from a condition file, optionally replayed as computed.

    The oracle sweep stored only the agent-reported configuration, so
    its computed counterpart is derived by applying the deterministic
    policy to the same audit -- the same replay used for the point
    estimates. Pairing computed estimates with agent-reported
    intervals would misdescribe both.
    """
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not replay:
        return rows
    wrapped = [{"multi": r} for r in rows]
    return [r["multi"] for r in
            apply_policy(wrapped, make_deterministic(1.0, gate=True))]


def supports_attribution(records: Sequence[dict]) -> bool:
    """Does this condition record the per-passage audit the cascade needs?"""
    return any(r.get("audit") for r in records)


def bootstrap_shares(
    records: Sequence[dict],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Dict[str, float]]:
    """Resample claims, re-attribute, and take percentile intervals.

    Returns, per stage, the point share and the 2.5/97.5 percentiles.

    ``attribute`` is a pure function of a single record, so a claim's
    stage does not depend on which other claims share the sample. Each
    claim is therefore attributed once and the resampling draws from
    the resulting labels, which gives the identical distribution to
    re-running the cascade per resample without rebuilding every audit
    ten thousand times.
    """
    point = attribute_run(records)
    n = len(records)
    rng = random.Random(seed)

    # None for a correct verdict; the stage name for a wrong one.
    labels: List[str | None] = []
    for record in records:
        if not (record.get("verdict") and record.get("gold")):
            labels.append(None)
        elif record["verdict"] == record["gold"]:
            labels.append(None)
        else:
            labels.append(attribute(record).stage)

    draws: Dict[str, List[float]] = {stage: [] for stage in STAGES}
    n_wrong: List[int] = []
    for _ in range(resamples):
        counts = {stage: 0 for stage in STAGES}
        wrong = 0
        for _ in range(n):
            stage = labels[rng.randrange(n)]
            if stage is not None:
                counts[stage] += 1
                wrong += 1
        n_wrong.append(wrong)
        for stage in STAGES:
            draws[stage].append(counts[stage] / wrong if wrong else 0.0)

    out: Dict[str, Dict[str, float]] = {}
    for stage in STAGES:
        series = sorted(draws[stage])
        out[stage] = {
            "share": point["shares"][stage],
            "count": point["counts"][stage],
            "lo": series[int(0.025 * resamples)],
            "hi": series[int(0.975 * resamples) - 1],
        }
    n_wrong.sort()
    out["_meta"] = {
        "n_scored": point["n_scored"],
        "n_wrong": point["n_wrong"],
        "wrong_lo": n_wrong[int(0.025 * resamples)],
        "wrong_hi": n_wrong[int(0.975 * resamples) - 1],
    }
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", type=Path,
        default=ROOT / "results" / "av250" / "B4.jsonl",
        help="Primary configuration under live retrieval.",
    )
    parser.add_argument(
        "--oracle", type=Path,
        default=ROOT / "results" / "av250_oracle" / "MAFCA.jsonl"
    )
    parser.add_argument("--b1", type=Path, default=ROOT / "results" / "av250" / "B1.jsonl")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "macros" / "attribution_numbers.tex"
    )
    args = parser.parse_args(argv)

    lines = [
        "% Generated by analysis/attribution_ci.py -- do not edit by hand.",
        f"% {args.resamples} bootstrap resamples over claims, seed {args.seed}.",
    ]
    emitted = 0

    for label, path, prefix, replay in (
        ("live", args.live, "CondErrLive", False),
        # The oracle file holds the agent-reported run; replay it so
        # both conditions describe the computed configuration.
        ("oracle", args.oracle, "CondErrOracle", True),
    ):
        if not path.exists():
            print(f"missing {path}, skipping {label}", file=sys.stderr)
            continue
        records = load(path, replay=replay)
        result = bootstrap_shares(
            records, resamples=args.resamples, seed=args.seed
        )
        meta = result["_meta"]
        print(
            f"\n{label}: {meta['n_wrong']} wrong of {meta['n_scored']} "
            f"(resampled range {meta['wrong_lo']}-{meta['wrong_hi']})"
        )
        for stage in STAGES:
            cell = result[stage]
            print(
                f"  {stage:14s} {100 * cell['share']:5.1f}%  "
                f"[{100 * cell['lo']:5.1f}, {100 * cell['hi']:5.1f}]  "
                f"n={cell['count']}"
            )
            tag = STAGE_TAG[stage]
            lines.append(
                f"\\newcommand{{\\{prefix}{tag}Lo}}{{{100 * cell['lo']:.1f}}}"
            )
            lines.append(
                f"\\newcommand{{\\{prefix}{tag}Hi}}{{{100 * cell['hi']:.1f}}}"
            )
            emitted += 2

    # B1: report whether attribution is even possible, rather than
    # silently omitting it.
    if args.b1.exists():
        b1 = load(args.b1)
        possible = supports_attribution(b1)
        print(
            f"\nB1 attribution possible: {possible} "
            f"({'audit recorded' if possible else 'no per-passage audit recorded'})"
        )
        lines.append(
            "\\newcommand{\\CondErrBoneAttributable}{"
            + ("yes" if possible else "no")
            + "}"
        )
        emitted += 1

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {emitted} macros -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
