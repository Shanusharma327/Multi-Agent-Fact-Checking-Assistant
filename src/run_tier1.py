"""Run the whole Tier 1 protocol end to end, resumably.

One command, so the experiment is reproducible as a unit rather than as a
sequence of remembered invocations:

    python run_tier1.py                    # full 400-claim protocol
    python run_tier1.py --n 20 --tag smoke # a quick rehearsal

Stages, in order. Each is skipped when its output already exists unless
``--force`` is given, so an interrupted run resumes rather than repeating
hours of inference:

1. **retrieve**  -- draw a label-balanced sample across dev and train and
   retrieve evidence once, into a passage cache.
2. **run**       -- replay every condition over that cache.
3. **derive**    -- compute B4 from MAFCA's stored audit, no inference.
4. **compare**   -- emit the result tables with intervals and tests.

Measured throughput on one local GPU at three workers (qwen2.5:3b):
retrieval ~13 s/claim, B0 ~6 s, B1 ~4 s, MAFCA ~7 s, B2 ~4 s, B3 ~7 s.
A 400-claim protocol is therefore roughly four and a half hours, of
which retrieval is about ninety minutes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fact_checker.locking import LockHeld, RunLock  # noqa: E402

# A progress line echoes the claim, and real-world claims carry currency
# symbols and accented names. A Windows console defaults to cp1252 and
# raises on the first one it cannot encode, which would otherwise end a
# long run over a cosmetic limitation of the terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass


ROOT = Path(__file__).resolve().parent

# B4 is derived from MAFCA rather than run, so it is absent here.
DEFAULT_CONDITIONS = ["B0", "B1", "MAFCA", "B2", "B3", "MAJORITY", "RANDOM"]


def run(args: Sequence[str], *, label: str) -> bool:
    """Run one stage, timing it and reporting rather than raising."""
    print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
    print("$ python " + " ".join(str(a) for a in args))
    started = time.perf_counter()
    result = subprocess.run([sys.executable, *[str(a) for a in args]], cwd=ROOT)
    elapsed = (time.perf_counter() - started) / 60
    if result.returncode != 0:
        print(f"  ! {label} failed (exit {result.returncode}) after {elapsed:.1f} min",
              file=sys.stderr)
        return False
    print(f"  {label} finished in {elapsed:.1f} min")
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=133, help="Claims per label.")
    parser.add_argument("--tag", default="av400", help="Names the cache and results.")
    parser.add_argument("--dataset", default="averitec")
    parser.add_argument("--splits", nargs="*", default=["dev", "train"],
                        help="Splits to pool. Provenance is recorded per claim.")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--limit", type=int, default=0,
                        help="Keep only the first N claims of the seeded draw.")
    parser.add_argument("--retrieval-workers", type=int, default=0,
                        help="Search concurrency; defaults to --workers minus one.")
    parser.add_argument("--force-lock", action="store_true",
                        help="Take the run lock even if another process holds it.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run stages whose output already exists.")
    parser.add_argument("--skip-retrieve", action="store_true",
                        help="Use the existing cache as-is.")
    args = parser.parse_args(argv)

    cache = ROOT / "cache" / f"{args.tag}.jsonl"
    out_dir = ROOT / "results" / args.tag

    # One protocol run per tag. Two concurrent runs corrupt each other's
    # outputs and, between them, rate-limited the search host into refusing
    # every request - which is how this lock came to exist.
    try:
        lock = RunLock(out_dir, name="tier1", force=args.force_lock).acquire()
    except LockHeld as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()

    # 1. Retrieval, once.
    #
    # "The cache file exists" is not the same as "retrieval is done". A
    # partial cache -- from an interrupted run, or seeded from a larger
    # draw -- would otherwise be treated as complete and the protocol
    # would silently evaluate a fraction of the intended sample. The
    # retrieve step is itself resumable, so the safe default is to run it
    # and let it skip what is already present.
    target = args.limit or (args.n * 3)
    cached = 0
    if cache.exists():
        cached = sum(
            1 for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    if args.skip_retrieve:
        print(f"Using existing cache as-is: {cache} ({cached} claims)")
    elif cached >= target and not args.force:
        print(f"Cache already holds {cached} of {target} claims: {cache}")
    else:
        if cached:
            print(f"Cache holds {cached} of {target} claims; retrieving the rest.")
        ok = run(
            ["run_conditions.py", "retrieve", "--dataset", args.dataset,
             "--splits", *args.splits, "--n", args.n, "--out", cache,
             "--workers", args.workers, "--seed", args.seed, "--resume"]
            + (["--limit", args.limit] if args.limit else [])
            + (["--retrieval-workers", args.retrieval_workers]
               if args.retrieval_workers else []),
            label=f"1/4  retrieve {args.n} x 3 claims into {cache.name}",
        )
        if not ok:
            return 1

    # 2. Conditions, replayed over that one cache.
    ok = run(
        ["run_conditions.py", "run", "--cache", cache,
         "--conditions", *args.conditions, "--out-dir", out_dir,
         "--workers", args.workers, "--seed", args.seed, "--resume"]
        + (["--limit", args.limit] if args.limit else []),
        label=f"2/4  run conditions: {', '.join(args.conditions)}",
    )
    if not ok:
        return 1

    # 3. B4, derived from MAFCA's audit so the two provably share it.
    mafca = out_dir / "MAFCA.jsonl"
    if mafca.exists():
        run(
            ["run_conditions.py", "derive", "--mafca", mafca,
             "--out", out_dir / "B4.jsonl"],
            label="3/4  derive B4 from MAFCA (no inference)",
        )

    # 4. Tables.
    ok = run(
        ["analysis/compare_conditions.py", "--dir", out_dir,
         "--resamples", args.resamples, "--seed", args.seed, "--reweight",
         "--json-out", out_dir / "summary.json"],
        label="4/4  compare conditions",
    )

    total = (time.perf_counter() - started) / 60
    print(f"\n{'=' * 78}")
    print(f"  Tier 1 complete in {total:.1f} min")
    print(f"  cache   {cache}")
    print(f"  results {out_dir}")
    print(f"{'=' * 78}")
    lock.release()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
