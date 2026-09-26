"""FEVER benchmark harness: multi-agent pipeline vs. single-LLM baseline.

Runs both systems over a label-balanced FEVER sample and scores them on
classification quality *and* calibration.

Typical use:

    python evaluate.py --download                 # fetch the dev split once
    python evaluate.py --n 20 --dry-run           # preview claims + cost
    python evaluate.py --n 20                     # run it
    python evaluate.py --score-only               # re-score existing results

Results stream to JSONL as they complete, so an interrupted run resumes with
``--resume`` instead of paying for the same claims twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fact_checker import Settings
from fact_checker.agents import run_baseline
from fact_checker.config import (
    DEFAULT_MODELS,
    PROVIDERS,
    default_provider,
    ollama_status,
)
from fact_checker.datasets import (
    DATASETS,
    DEFAULT_SPLITS,
    SPLIT_CHOICES,
    BenchmarkClaim,
    UnknownDatasetError,
    download_benchmark,
    label_distribution,
    load_benchmark,
    sample_balanced,
)
from fact_checker.datasets.averitec import AVeriTeCDownloadError
from fact_checker.datasets.fever import FeverDownloadError
from fact_checker.llm import AgentError, ConfigurationError
from fact_checker.metrics import (
    Scorecard,
    score_subset,
    format_confusion,
    format_scorecard,
    score_system,
)
from fact_checker.pipeline import run_multi_agent

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for _noisy in ("ddgs", "primp", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

logger = logging.getLogger("evaluate")

DEFAULT_OUT = Path("results/fever_run.jsonl")

# Per-claim token estimate: 3 pipeline calls + 1 baseline call.
_EST_INPUT_TOKENS = 4_450
_EST_OUTPUT_TOKENS = 2_200

# USD per million tokens (input, output).
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_write_lock = threading.Lock()


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def estimate_cost(
    model: str, n_claims: int, provider: str = "anthropic"
) -> Optional[float]:
    """Rough USD estimate for a run.

    Returns 0.0 for local inference (electricity aside, it is free) and None
    for a hosted model whose pricing is not in the table.
    """
    if provider == "ollama":
        return 0.0
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    price_in, price_out = pricing
    per_claim = (
        _EST_INPUT_TOKENS / 1e6 * price_in + _EST_OUTPUT_TOKENS / 1e6 * price_out
    )
    return per_claim * n_claims


def confirm(prompt: str) -> bool:
    """Ask for interactive confirmation; declines automatically when piped."""
    if not sys.stdin.isatty():
        print(f"{prompt} [non-interactive stdin -> declining; pass --yes to proceed]")
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def evaluate_claim(claim: BenchmarkClaim, settings: Settings) -> dict:
    """Run both systems over one claim and return a result record.

    Failures are captured into the record rather than raised, so a single bad
    claim cannot abort a long, already-paid-for run.
    """
    record: dict = {
        "id": str(claim.id),
        "claim": claim.claim,
        "gold": claim.label,
        "gold_pages": list(getattr(claim, "gold_pages", [])),
        "dataset": claim.dataset,
        "multi": None,
        "baseline": None,
        "errors": [],
    }

    try:
        started = time.perf_counter()
        result = run_multi_agent(claim.claim, settings)
        record["multi"] = {
            "verdict": result.adjudication.final_verdict,
            "confidence": result.adjudication.confidence,
            "cited_sources": result.adjudication.cited_sources,
            "n_evidence": len(result.scout.evidence),
            "queries": result.scout.queries,
            "audit_tally": result.audit.tally(),
            "reasoning": result.adjudication.detailed_reasoning,
            "seconds": time.perf_counter() - started,
            "warnings": result.warnings,
        }
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"multi-agent: {exc}")

    try:
        baseline = run_baseline(claim.claim, settings)
        record["baseline"] = {
            "verdict": baseline.adjudication.final_verdict,
            "confidence": baseline.adjudication.confidence,
            "reasoning": baseline.adjudication.detailed_reasoning,
            "seconds": baseline.seconds,
        }
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"baseline: {exc}")

    return record


def load_existing(path: Path) -> Dict[str, dict]:
    """Read previously written results, keyed by claim id."""
    if not path.exists():
        return {}
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                records[str(record["id"])] = record
    return records


def run_benchmark(
    claims: List[BenchmarkClaim],
    settings: Settings,
    out_path: Path,
    *,
    workers: int = 3,
) -> List[dict]:
    """Evaluate every claim, appending each result to ``out_path`` as it lands."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    total = len(claims)
    completed = 0

    def write(record: dict) -> None:
        with _write_lock:
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_claim, claim, settings): claim for claim in claims
        }
        for future in as_completed(futures):
            claim = futures[future]
            completed += 1
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the run alive.
                record = {
                    "id": str(claim.id),
                    "claim": claim.claim,
                    "gold": claim.label,
                    "multi": None,
                    "baseline": None,
                    "errors": [f"unexpected: {exc}"],
                }
            results.append(record)
            write(record)

            multi = (record.get("multi") or {}).get("verdict", "ERR")
            base = (record.get("baseline") or {}).get("verdict", "ERR")
            flag = "OK " if multi == record["gold"] else "MISS"
            print(
                f"  [{completed:>4}/{total}] {flag} gold={record['gold']:<16} "
                f"multi={multi:<16} base={base:<16} {record['claim'][:52]}"
            )

    return results


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_records(records: List[dict]) -> Tuple[Optional[Scorecard], Optional[Scorecard], dict]:
    """Score both systems over the subset of records each one completed.

    Each system is scored on the claims *it* completed, and the counts are
    reported, so a few failures on one side cannot silently shrink the other's
    denominator.
    """
    multi_gold, multi_pred, multi_conf, multi_secs = [], [], [], []
    base_gold, base_pred, base_conf, base_secs = [], [], [], []

    for record in records:
        gold = record.get("gold")
        if not gold:
            continue
        multi = record.get("multi")
        if multi and multi.get("verdict"):
            multi_gold.append(gold)
            multi_pred.append(multi["verdict"])
            multi_conf.append(float(multi.get("confidence", 0.0)))
            multi_secs.append(float(multi.get("seconds", 0.0)))
        baseline = record.get("baseline")
        if baseline and baseline.get("verdict"):
            base_gold.append(gold)
            base_pred.append(baseline["verdict"])
            base_conf.append(float(baseline.get("confidence", 0.0)))
            base_secs.append(float(baseline.get("seconds", 0.0)))

    multi_card = (
        score_system("multi-agent (grounded)", multi_gold, multi_pred, multi_conf, multi_secs)
        if multi_gold
        else None
    )
    base_card = (
        score_system("baseline (ungrounded)", base_gold, base_pred, base_conf, base_secs)
        if base_gold
        else None
    )

    # Agreement statistics over claims both systems completed.
    paired = [
        r
        for r in records
        if (r.get("multi") or {}).get("verdict") and (r.get("baseline") or {}).get("verdict")
    ]
    agree = sum(1 for r in paired if r["multi"]["verdict"] == r["baseline"]["verdict"])
    multi_only = sum(
        1
        for r in paired
        if r["multi"]["verdict"] == r["gold"] and r["baseline"]["verdict"] != r["gold"]
    )
    base_only = sum(
        1
        for r in paired
        if r["baseline"]["verdict"] == r["gold"] and r["multi"]["verdict"] != r["gold"]
    )
    extra = {
        "paired": len(paired),
        "agree": agree,
        "multi_only_correct": multi_only,
        "baseline_only_correct": base_only,
        "errors": sum(1 for r in records if r.get("errors")),
    }
    return multi_card, base_card, extra


def print_report(records: List[dict], dataset: str = "fever") -> None:
    """Print the full comparison report to stdout."""
    multi_card, base_card, extra = score_records(records)

    print("\n" + "=" * 74)
    print("FEVER BENCHMARK  ·  multi-agent pipeline vs. single-LLM baseline")
    print("=" * 74)

    for card in (multi_card, base_card):
        if card is None:
            continue
        print()
        print(format_scorecard(card))
        print("    confusion:")
        print(format_confusion(card))

    if multi_card and base_card:
        print("\n" + "-" * 74)
        print("  HEAD-TO-HEAD")
        print(
            f"    accuracy            "
            f"{multi_card.accuracy:6.1%}  vs {base_card.accuracy:6.1%}   "
            f"({multi_card.accuracy - base_card.accuracy:+.1%})"
        )
        print(
            f"    macro F1            "
            f"{multi_card.macro_f1:6.1%}  vs {base_card.macro_f1:6.1%}   "
            f"({multi_card.macro_f1 - base_card.macro_f1:+.1%})"
        )
        print(
            f"    ECE                 "
            f"{multi_card.ece:6.3f}  vs {base_card.ece:6.3f}   "
            f"({multi_card.ece - base_card.ece:+.3f})"
        )
        print(
            f"    overconfidence      "
            f"{multi_card.overconfidence:+6.1%}  vs {base_card.overconfidence:+6.1%}"
        )
        print(f"    agreed on           {extra['agree']}/{extra['paired']} claims")
        print(f"    only multi correct  {extra['multi_only_correct']}")
        print(f"    only baseline right {extra['baseline_only_correct']}")

    # --- decidable (SUPPORTS/REFUTES) vs. NEI, reported separately ---
    paired_rows = [
        (
            r["gold"],
            r["multi"]["verdict"],
            float(r["multi"].get("confidence", 0.0)),
            r["baseline"]["verdict"],
            float(r["baseline"].get("confidence", 0.0)),
        )
        for r in records
        if (r.get("multi") or {}).get("verdict")
        and (r.get("baseline") or {}).get("verdict")
    ]
    if paired_rows:
        gold = [row[0] for row in paired_rows]
        print("\n" + "-" * 74)
        print("  REPORTED BY LABEL GROUP")
        print(
            "    NOT_ENOUGH_INFO is separated from the headline because on FEVER it\n"
            "    is corpus-relative -- 'unverifiable from a 2017 Wikipedia snapshot',\n"
            "    not 'unverifiable in principle'. A single blended accuracy would\n"
            "    conflate system quality with benchmark construction."
        )
        for name, pred_i, conf_i in (("multi-agent", 1, 2), ("baseline   ", 3, 4)):
            pred = [row[pred_i] for row in paired_rows]
            conf = [row[conf_i] for row in paired_rows]
            decidable = score_subset(name, gold, pred, conf, ["SUPPORTS", "REFUTES"])
            nei = score_subset(name, gold, pred, conf, ["NOT_ENOUGH_INFO"])
            print(
                f"    {name}  decidable (S/R): {decidable.accuracy:6.1%} "
                f"(n={decidable.n:>3})   |   NEI only: {nei.accuracy:6.1%} "
                f"(n={nei.n:>3})"
            )

    if extra["errors"]:
        print(f"\n  NOTE: {extra['errors']} claim(s) recorded an error.")

    if dataset == "fever":
        print(
            "\n  Caveat: FEVER labels are defined against a 2017 Wikipedia dump, while\n"
            "  this pipeline retrieves from the open web. NOT_ENOUGH_INFO recall is\n"
            "  expected to be depressed -- live search can genuinely verify claims\n"
            "  FEVER marks unverifiable. FEVER also predates these models, so the\n"
            "  ungrounded baseline may be partly recalling the benchmark itself.\n"
            "  Cross-check: python evaluate.py --dataset averitec"
        )
    else:
        print(
            "\n  Note: AVeriTeC claims are real-world and open-web, matching this\n"
            "  system's target distribution, but the split is label-imbalanced and\n"
            "  small. Claims carrying its fourth label are excluded, not remapped."
        )
    print("=" * 74)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the multi-agent pipeline against the single-LLM baseline "
            "on FEVER (comparability) or AVeriTeC (validity)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10, help="Claims per label (x3 labels).")
    parser.add_argument(
        "--dataset",
        default="fever",
        choices=list(DATASETS),
        help="Benchmark corpus. 'averitec' is real-world, open-web, 2023.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split name. Defaults to the dataset's dev split.",
    )
    parser.add_argument("--seed", type=int, default=20240101, help="Sampling seed.")
    parser.add_argument(
        "--provider",
        default=None,
        choices=list(PROVIDERS),
        help="Model backend. Defaults to anthropic when a key is set, else ollama.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to the provider's default (e.g. qwen2.5:7b).",
    )
    parser.add_argument("--ollama-host", default=None, help="Ollama base URL.")
    parser.add_argument("--workers", type=int, default=3, help="Parallel claims.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Results JSONL.")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache directory. Defaults per dataset (data/<dataset>).",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the split and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the sample and cost, then exit."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip claims already in --out."
    )
    parser.add_argument(
        "--score-only", action="store_true", help="Re-score --out without running."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    provider = args.provider or default_provider()
    model = args.model or DEFAULT_MODELS[provider]
    dataset = args.dataset
    split = args.split or DEFAULT_SPLITS[dataset]
    cache_dir = str(args.cache_dir) if args.cache_dir else ""

    if split not in SPLIT_CHOICES[dataset]:
        print(
            f"ERROR: {dataset} has no split {split!r}. "
            f"Choose from: {', '.join(SPLIT_CHOICES[dataset])}",
            file=sys.stderr,
        )
        return 1

    if args.download:
        try:
            path = download_benchmark(dataset, split, cache_dir)
        except (FeverDownloadError, AVeriTeCDownloadError, UnknownDatasetError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(
            f"Downloaded {dataset} {split} -> {path} "
            f"({path.stat().st_size / 1e6:.1f} MB)"
        )
        return 0

    if args.score_only:
        existing = load_existing(args.out)
        if not existing:
            print(f"No results found at {args.out}", file=sys.stderr)
            return 1
        print(f"Scoring {len(existing)} result(s) from {args.out}")
        print_report(list(existing.values()), dataset)
        return 0

    # ---- sample ----
    try:
        population, load_report = load_benchmark(dataset, split, cache_dir)
    except (FeverDownloadError, AVeriTeCDownloadError, UnknownDatasetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    claims = sample_balanced(population, args.n, seed=args.seed)
    print(
        f"{dataset} {split}: {len(population):,} usable claims "
        f"(of {load_report.get('total', len(population)):,}) -> "
        f"sampled {len(claims)} ({label_distribution(claims)})"
    )
    if load_report.get("conflicting"):
        # Never let an exclusion pass silently.
        print(
            f"  note: {load_report['conflicting']} claim(s) excluded for carrying "
            "AVeriTeC's 'Conflicting Evidence/Cherrypicking' label, which has no "
            "three-way equivalent."
        )
    dist = label_distribution(claims)
    if len(set(dist.values())) > 1:
        print(
            f"  note: sample is NOT balanced {dist} - the population lacks enough "
            "claims in at least one label."
        )

    already = load_existing(args.out) if args.resume else {}
    if already:
        before = len(claims)
        claims = [c for c in claims if str(c.id) not in already]
        print(f"Resuming: {before - len(claims)} already done, {len(claims)} remaining.")

    if args.dry_run:
        print("\nSample preview:")
        for claim in claims[:15]:
            print(f"  [{claim.label:<16}] {claim.claim}")
        if len(claims) > 15:
            print(f"  ... and {len(claims) - 15} more")
        cost = estimate_cost(model, len(claims), provider)
        if provider == "ollama":
            print(
                f"\nBackend: ollama/{model} - local inference, no API cost."
            )
        elif cost is not None:
            print(f"\nEstimated cost on {model}: ~${cost:.2f} "
                  f"({len(claims)} claims x 4 LLM calls)")
        return 0

    if not claims:
        print("Nothing to run.")
        if already:
            print_report(list(already.values()), dataset)
        return 0

    settings = Settings(
        provider=provider,
        model=model,
        **({"ollama_host": args.ollama_host} if args.ollama_host else {}),
    )
    if not settings.is_configured:
        print(
            "ERROR: no API key. Set ANTHROPIC_API_KEY in your environment or .env, "
            "or pass --provider ollama to run locally.",
            file=sys.stderr,
        )
        return 1

    if settings.is_local:
        reachable, installed = ollama_status(settings.ollama_host)
        if not reachable:
            print(
                f"ERROR: no Ollama server at {settings.ollama_host}. "
                f"Install from ollama.com, then: ollama pull {model}",
                file=sys.stderr,
            )
            return 1
        if installed and model not in installed:
            print(
                f"ERROR: model {model!r} is not pulled. Run: ollama pull {model}\n"
                f"Installed: {', '.join(installed) or '(none)'}",
                file=sys.stderr,
            )
            return 1

    # ---- confirm spend ----
    cost = estimate_cost(model, len(claims), provider)
    if settings.is_local:
        print(f"\nRunning {len(claims)} claims locally on {model} - no API cost.")
    elif cost is not None:
        print(
            f"\nAbout to run {len(claims)} claims x 4 LLM calls on {model}.\n"
            f"Estimated cost: ~${cost:.2f} (rough; actual depends on evidence length)."
        )
        if not args.yes and not confirm("Proceed?"):
            print("Aborted.")
            return 1

    print(f"\nRunning with {args.workers} worker(s) -> {args.out}\n")
    started = time.perf_counter()
    fresh = run_benchmark(claims, settings, args.out, workers=args.workers)
    elapsed = time.perf_counter() - started

    combined = list(already.values()) + fresh
    print(f"\nCompleted {len(fresh)} claim(s) in {elapsed:.1f}s.")
    print_report(combined, dataset)
    print(f"\nRaw results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
