"""Run experimental conditions over a fixed passage cache.

Separating retrieval from everything downstream is what makes the condition
comparisons controlled. Retrieval happens once, into a cache; every
condition then replays that cache, so two conditions differ in the thing
under test and in nothing else. The run aborts if that ever stops being
true.

    # 1. build a cache from an existing benchmark run (no retrieval)
    python run_conditions.py cache --from-results results/averitec_n60_v2.jsonl \\
        --out cache/averitec_n60.jsonl

    # 2. or retrieve fresh for a new sample
    python run_conditions.py retrieve --dataset averitec --n 133 \\
        --out cache/averitec_n400.jsonl

    # 3. run conditions over a cache
    python run_conditions.py run --cache cache/averitec_n60.jsonl \\
        --conditions B0 B1 MAFCA B4 --out-dir results/conditions

Per-condition results land in ``<out-dir>/<condition>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fact_checker import Settings

# A progress line echoes the claim, and real-world claims carry currency
# symbols and accented names. A Windows console defaults to cp1252 and
# raises on the first one it cannot encode, which would otherwise end a
# long run over a cosmetic limitation of the terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass

from fact_checker.cache import (
    CachedClaim,
    CachedPassage,
    PassageCache,
    assert_identical_evidence,
    utc_now,
)
from fact_checker.conditions import (
    CONDITIONS,
    condition_majority,
    condition_random,
)
from fact_checker.config import DEFAULT_MODELS, PROVIDERS, default_provider, ollama_status
from fact_checker.locking import LockHeld, RunLock
from fact_checker.datasets import (
    DATASETS,
    DEFAULT_SPLITS,
    BenchmarkClaim,
    label_distribution,
    load_benchmark,
    sample_balanced,
)
from fact_checker.search import SearchUnavailable, gather_evidence

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for _noisy in ("ddgs", "primp", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("conditions")
_write_lock = threading.Lock()


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Build settings, honouring FACTCHECK_MODEL as the library does."""
    provider = args.provider or default_provider()
    model = (
        args.model
        or os.getenv("FACTCHECK_MODEL", "").strip()
        or DEFAULT_MODELS[provider]
    )
    return Settings(
        provider=provider,
        model=model,
        **({"aggregation_beta": args.beta} if getattr(args, "beta", None) else {}),
    )


# ---------------------------------------------------------------------------
# cache  -- build from an existing run
# ---------------------------------------------------------------------------

def cmd_cache(args: argparse.Namespace) -> int:
    """Turn a benchmark results file into a replayable passage cache."""
    source = Path(args.from_results)
    if not source.exists():
        print(f"No results file at {source}", file=sys.stderr)
        return 1

    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cache = PassageCache.from_results(records, retrieved_at=args.retrieved_at or "")

    if args.split:
        # Runs predating provenance capture recorded no split. Stamping it
        # from the known run configuration is recovering a fact, not
        # inventing one -- but it is only safe when the whole run used one
        # split, which is why it must be passed explicitly.
        for entry in cache:
            entry.split = args.split
    if not len(cache):
        print(
            f"{source} carries no stored passages. It predates trace capture; "
            "re-run evaluate.py or use `retrieve`.",
            file=sys.stderr,
        )
        return 1

    cache.save(Path(args.out))
    report_cache(cache, Path(args.out))
    if not args.retrieved_at and any(not e.retrieved_at for e in cache):
        print(
            "\n  note: the source run recorded no retrieval timestamp, so "
            "entries carry none.\n  Pass --retrieved-at to stamp them with the "
            "run's known date."
        )
    return 0


# ---------------------------------------------------------------------------
# retrieve -- fresh retrieval into a cache
# ---------------------------------------------------------------------------

def _retrieve_one(claim: BenchmarkClaim, settings: Settings) -> CachedClaim:
    """Run Agent 1 for one claim and record what it found."""
    from fact_checker.agents.scout import run_scout

    fact_check_url = str(getattr(claim, "fact_checking_article", "") or "")
    queries: List[str] = []
    passages: List[CachedPassage] = []
    warnings: List[str] = []

    try:
        report, warns = run_scout(claim.claim, settings, exclude_url=fact_check_url)
        queries = list(report.queries)
        warnings = list(warns)
        passages = [
            CachedPassage(
                source_id=item.source_id,
                url=item.url,
                title=item.title,
                snippet=item.snippet,
                query=item.query,
            )
            for item in report.evidence
        ]
    except Exception as exc:  # noqa: BLE001 - a dead claim must not kill the run.
        warnings.append(f"retrieval failed: {exc}")

    return CachedClaim(
        claim_id=str(claim.id),
        dataset=claim.dataset,
        split=str(getattr(claim, "split", "") or ""),
        claim=claim.claim,
        gold=claim.label,
        gold_urls=list(getattr(claim, "gold_urls", []) or []),
        fact_check_url=fact_check_url,
        queries=queries,
        passages=passages,
        retrieved_at=utc_now(),
        retrieval_warnings=warnings,
    )


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Draw a sample and retrieve evidence for it, once, into a cache."""
    settings = resolve_settings(args)
    if settings.is_local:
        reachable, installed = ollama_status(settings.ollama_host)
        if not reachable:
            print(f"ERROR: no Ollama server at {settings.ollama_host}", file=sys.stderr)
            return 1
        if installed and settings.model not in installed:
            print(
                f"ERROR: model {settings.model!r} not pulled. Installed: "
                f"{', '.join(installed) or '(none)'}",
                file=sys.stderr,
            )
            return 1

    splits = args.splits or [DEFAULT_SPLITS[args.dataset]]
    population: List[BenchmarkClaim] = []
    for split in splits:
        # The loaders record provenance on each claim, so a sample topped up
        # across splits stays interpretable without the caller stamping it.
        claims, report = load_benchmark(args.dataset, split, args.cache_dir or "")
        population.extend(claims)
        print(
            f"{args.dataset} {split}: {report.get('kept', len(claims)):,} usable "
            f"of {report.get('total', 0):,}"
            + (
                f"  ({report['conflicting']} excluded for the fourth label)"
                if report.get("conflicting")
                else ""
            )
        )

    sample = sample_balanced(population, args.n, seed=args.seed)
    if args.limit and args.limit < len(sample):
        # Truncating the shuffled draw keeps the sample a deterministic
        # prefix of the same seeded set, so claims already retrieved under
        # a larger --n are reused rather than discarded. The draw is
        # shuffled, so a prefix is close to balanced but not exactly so;
        # the realised counts are printed below and belong in the paper.
        sample = sample[: args.limit]
    distribution = label_distribution(sample)
    print(f"\nsampled {len(sample)} claims: {distribution}")
    if len(set(distribution.values())) > 1:
        print("  note: NOT balanced - the population lacks claims in some label.")

    by_split: Dict[str, int] = {}
    for claim in sample:
        key = str(getattr(claim, "split", "") or "(unknown)")
        by_split[key] = by_split.get(key, 0) + 1
    print(f"  split provenance: {by_split}")

    if args.dry_run:
        for claim in sample[:12]:
            print(f"  [{claim.label:<16}] {claim.claim[:76]}")
        if len(sample) > 12:
            print(f"  ... and {len(sample) - 12} more")
        return 0

    out = Path(args.out)
    try:
        lock = RunLock(out, name="retrieve", force=args.force_lock).acquire()
    except LockHeld as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cache = PassageCache.load(out) if (out.exists() and args.resume) else PassageCache()
    todo = [c for c in sample if str(c.id) not in cache]
    if args.resume and len(todo) < len(sample):
        print(f"  resuming: {len(sample) - len(todo)} already cached, {len(todo)} to go")

    # Retrieval is throttled by the search host, not by local compute, so it
    # gets its own lower worker count; inference concurrency is unaffected.
    retrieval_workers = args.retrieval_workers or max(1, args.workers - 1)

    print(f"\nRetrieving with {retrieval_workers} worker(s) -> {out}\n")
    started = time.perf_counter()
    done = 0
    stalled = False
    with ThreadPoolExecutor(max_workers=retrieval_workers) as pool:
        futures = {pool.submit(_retrieve_one, c, settings): c for c in todo}
        for future in as_completed(futures):
            try:
                entry = future.result()
            except SearchUnavailable as exc:
                # The service is down, not this claim. Stop asking.
                print(f"\n  HALTING: {exc}")
                stalled = True
                break
            done += 1
            with _write_lock:
                cache.add(entry)
                if done % 10 == 0 or done == len(todo):
                    cache.save(out)  # checkpoint, so a crash loses little
            print(
                f"  [{done:>4}/{len(todo)}] {len(entry.passages)} passage(s)  "
                f"{entry.claim[:58]}"
            )

    cache.save(out)
    lock.release()
    elapsed = time.perf_counter() - started
    print(f"\nRetrieved {len(todo)} claim(s) in {elapsed / 60:.1f} min")
    report_cache(cache, out)
    if stalled:
        print(
            "\n  The search service stopped answering and retrieval halted "
            f"early with {len(cache)} of {len(sample)} claims cached.\n"
            "  Nothing is lost: re-run the same command with --resume once "
            "the service recovers."
        )
        return 2
    return 0


def report_cache(cache: PassageCache, path: Path) -> None:
    """Print the counts a reader needs to interpret this cache."""
    summary = cache.summary()
    print(f"\nCache: {path}")
    print(f"  claims            {summary['claims']}")
    print(f"  passages          {summary['total_passages']} "
          f"(mean {summary['mean_passages']:.2f}/claim)")
    print(f"  empty evidence    {summary['empty_evidence']}")
    print(f"  labels            {summary['labels']}")
    print(f"  splits            {summary['splits']}")
    first, last = summary["retrieved_between"]
    print(f"  retrieved between {first or '(unstamped)'} .. {last or '(unstamped)'}")


# ---------------------------------------------------------------------------
# run -- replay conditions over a cache
# ---------------------------------------------------------------------------

def _write_jsonl(records: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    """Run the requested conditions over one cache and verify identity."""
    cache = PassageCache.load(Path(args.cache))
    entries = list(cache)
    if args.limit:
        entries = entries[: args.limit]
    report_cache(cache.subset(e.claim_id for e in entries), Path(args.cache))

    settings = resolve_settings(args)
    print(f"\nBackend: {settings.provider}/{settings.model}")
    if settings.is_local:
        reachable, installed = ollama_status(settings.ollama_host)
        if not reachable:
            print(f"ERROR: no Ollama server at {settings.ollama_host}", file=sys.stderr)
            return 1
        if installed and settings.model not in installed:
            print(f"ERROR: model {settings.model!r} not pulled.", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = RunLock(out_dir, name="conditions", force=args.force_lock).acquire()
    except LockHeld as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    all_records: Dict[str, List[dict]] = {}

    for name in args.conditions:
        if name in ("MAJORITY", "RANDOM"):
            records = (
                condition_majority(entries)
                if name == "MAJORITY"
                else condition_random(entries, seed=args.seed)
            )
            _write_jsonl(records, out_dir / f"{name}.jsonl")
            all_records[name] = records
            print(f"\n{name}: {len(records)} record(s), no inference")
            continue

        if name not in CONDITIONS:
            print(f"ERROR: unknown condition {name!r}. "
                  f"Choose from: {', '.join(sorted(CONDITIONS) )}, MAJORITY, RANDOM",
                  file=sys.stderr)
            return 1

        target = out_dir / f"{name}.jsonl"
        done: Dict[str, dict] = {}
        if args.resume and target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    done[str(record.get("id"))] = record

        todo = [e for e in entries if e.claim_id not in done]
        print(f"\n{name}: {len(todo)} to run, {len(done)} cached")
        if not todo:
            all_records[name] = list(done.values())
            continue

        function = CONDITIONS[name]
        records: List[dict] = list(done.values())
        started = time.perf_counter()
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(function, e, settings): e for e in todo}
            for future in as_completed(futures):
                entry = futures[future]
                completed += 1
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - keep the run alive.
                    record = {
                        "id": entry.claim_id,
                        "condition": name,
                        "gold": entry.gold,
                        "errors": [f"unexpected: {exc}"],
                    }
                records.append(record)
                with _write_lock:
                    _write_jsonl(records, target)
                flag = "OK " if record.get("verdict") == entry.gold else "MISS"
                print(
                    f"  [{completed:>4}/{len(todo)}] {flag} "
                    f"gold={entry.gold:<16} pred={record.get('verdict', 'ERR'):<16} "
                    f"{entry.claim[:46]}"
                )

        elapsed = time.perf_counter() - started
        per_claim = elapsed / len(todo) if todo else 0.0
        print(f"  {name}: {len(todo)} claim(s) in {elapsed / 60:.1f} min "
              f"({per_claim:.1f}s/claim at {args.workers} workers)")
        all_records[name] = records

    # The guarantee the whole design rests on.
    checked = assert_identical_evidence(all_records)
    print(f"\nEvidence identity verified across conditions on {checked} claim(s).")
    print(f"Results -> {out_dir}")
    lock.release()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_derive(args: argparse.Namespace) -> int:
    """Derive B4 from MAFCA's records by replaying the aggregation in code.

    B4 differs from MAFCA only in how the audit becomes a verdict, and that
    step is a pure function of the audit. Re-running the two model calls to
    obtain it would cost inference and, worse, would let the audits drift
    apart -- so the two conditions would no longer be comparable on the one
    axis the ablation is about. Deriving guarantees they share the audit.
    """
    source = Path(args.mafca)
    if not source.exists():
        print(f"No MAFCA results at {source}", file=sys.stderr)
        return 1

    from fact_checker.aggregation import aggregate, consistent_citations, supporting_ids
    from fact_checker.schemas import Alignment, AuditReport

    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    derived: List[dict] = []
    for record in records:
        if not record.get("verdict"):
            continue
        audit = AuditReport(
            alignments=[
                Alignment(
                    source_id=str(entry.get("source_id", "")),
                    verdict=entry.get("verdict", "NOT_ENOUGH_INFO"),
                    justification=str(entry.get("justification", "")),
                )
                for entry in (record.get("audit") or [])
                if entry.get("source_id")
            ]
        )
        trace = aggregate(audit, beta=args.beta)
        valid = set(record.get("evidence_ids") or [])
        citations = consistent_citations(
            record.get("cited_sources") or [], audit, trace.verdict, valid
        )
        if not citations:
            citations = [i for i in supporting_ids(audit, trace.verdict) if i in valid]

        derived.append(
            {
                **record,
                "condition": "B4",
                "verdict": trace.verdict,
                "confidence": trace.confidence,
                "cited_sources": list(dict.fromkeys(citations)),
                "aggregation_trace": trace.model_dump(),
                # No model call: the two that produced the audit are MAFCA's
                # and are already counted there.
                "llm_calls": record.get("llm_calls", 2),
                "derived_from": str(source),
            }
        )

    out = Path(args.out)
    _write_jsonl(derived, out)
    changed = sum(
        1 for d, r in zip(derived, records) if d["verdict"] != r.get("verdict")
    )
    print(f"Derived B4 for {len(derived)} claim(s) -> {out}")
    print(f"  verdict changed on {changed} claim(s) relative to MAFCA")
    return 0


def cmd_oracle_cache(args: argparse.Namespace) -> int:
    """Build a cache whose passages are the corpus' annotated gold evidence.

    This is the oracle-retrieval condition: it substitutes the evidence a
    human fact-checker actually used for whatever search returned, which
    scores verification and aggregation independently of retrieval. Since
    the error attribution puts most failures on retrieval, the oracle is
    what says how much of that is recoverable.

    It is an upper bound, not a deployable mode -- at inference time no
    gold evidence exists -- and the paper reports it as such. Only claims
    present in the live cache are included, so the oracle and live
    conditions are compared on exactly the same claims.
    """
    live = PassageCache.load(Path(args.cache))

    by_id = {}
    for split in args.splits or ["dev", "train"]:
        claims, _ = load_benchmark(args.dataset, split, args.cache_dir or "")
        for claim in claims:
            by_id[str(claim.id)] = claim

    oracle_cache = PassageCache()
    missing = 0
    for entry in live:
        claim = by_id.get(entry.claim_id)
        passages = list(getattr(claim, "gold_evidence", []) or []) if claim else []
        if not passages:
            # No annotated evidence: carried through as an empty set rather
            # than dropped, so the claim set stays identical to the live one.
            missing += 1
        urls = list(getattr(claim, "gold_urls", []) or []) if claim else []
        oracle_cache.add(
            CachedClaim(
                claim_id=entry.claim_id,
                dataset=entry.dataset,
                split=entry.split,
                claim=entry.claim,
                gold=entry.gold,
                gold_urls=entry.gold_urls,
                fact_check_url=entry.fact_check_url,
                queries=["(oracle retrieval: annotated gold evidence)"],
                passages=[
                    CachedPassage(
                        source_id=f"doc_{index}",
                        url=urls[index - 1] if index - 1 < len(urls) else "",
                        title="Annotated evidence",
                        snippet=text,
                        query="(oracle)",
                    )
                    for index, text in enumerate(passages, start=1)
                ],
                retrieved_at=entry.retrieved_at,
                retrieval_warnings=["oracle retrieval: no live search performed"],
            )
        )

    oracle_cache.save(Path(args.out))
    report_cache(oracle_cache, Path(args.out))
    if missing:
        print(f"\n  note: {missing} claim(s) carry no annotated evidence and "
              "have an empty oracle set.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", default=None, choices=list(PROVIDERS))
    common.add_argument("--model", default=None)
    common.add_argument("--workers", type=int, default=3)
    common.add_argument("--seed", type=int, default=20240101)

    p_cache = sub.add_parser("cache", help="Build a cache from a results file.")
    p_cache.add_argument("--from-results", required=True)
    p_cache.add_argument("--out", required=True)
    p_cache.add_argument("--retrieved-at", default="",
                         help="ISO-8601 stamp for runs that recorded none.")
    p_cache.add_argument("--split", default="",
                         help="Split to stamp, for runs predating provenance capture.")

    p_ret = sub.add_parser("retrieve", parents=[common],
                           help="Sample claims and retrieve evidence into a cache.")
    p_ret.add_argument("--dataset", default="averitec", choices=list(DATASETS))
    p_ret.add_argument("--splits", nargs="*", default=None,
                       help="Splits to pool, e.g. dev train. Provenance is recorded.")
    p_ret.add_argument("--n", type=int, default=133, help="Claims per label.")
    p_ret.add_argument("--cache-dir", default=None)
    p_ret.add_argument("--out", required=True)
    p_ret.add_argument("--resume", action="store_true")
    p_ret.add_argument(
        "--retrieval-workers", type=int, default=0,
        help="Concurrency for search only. Defaults to --workers minus one, "
             "because the search host throttles long before local compute does.")
    p_ret.add_argument(
        "--force-lock", action="store_true",
        help="Take the run lock even if another process holds it.")
    p_ret.add_argument("--dry-run", action="store_true")
    p_ret.add_argument(
        "--limit", type=int, default=0,
        help="Keep only the first N claims of the seeded draw, so a smaller "
             "sample reuses what a larger one already retrieved.")

    p_run = sub.add_parser("run", parents=[common],
                           help="Replay conditions over a cache.")
    p_run.add_argument("--cache", required=True)
    p_run.add_argument("--conditions", nargs="+", required=True)
    p_run.add_argument("--out-dir", default="results/conditions")
    p_run.add_argument("--limit", type=int, default=0, help="First N claims only.")
    p_run.add_argument("--beta", type=float, default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument(
        "--force-lock", action="store_true",
        help="Take the run lock even if another process holds it.")

    p_orc = sub.add_parser(
        "oracle-cache",
        help="Build a cache of annotated gold evidence for the same claims.")
    p_orc.add_argument("--cache", required=True, help="The live cache to mirror.")
    p_orc.add_argument("--dataset", default="averitec", choices=list(DATASETS))
    p_orc.add_argument("--splits", nargs="*", default=None)
    p_orc.add_argument("--cache-dir", default=None)
    p_orc.add_argument("--out", required=True)

    p_der = sub.add_parser("derive", help="Derive B4 from MAFCA records, no inference.")
    p_der.add_argument("--mafca", required=True, help="MAFCA results JSONL.")
    p_der.add_argument("--out", required=True)
    p_der.add_argument("--beta", type=float, default=1.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "cache": cmd_cache,
        "retrieve": cmd_retrieve,
        "run": cmd_run,
        "derive": cmd_derive,
        "oracle-cache": cmd_oracle_cache,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
