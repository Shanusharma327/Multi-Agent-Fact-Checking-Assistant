"""Command-line fact checker: verify one claim without opening the UI.

    python check.py "Humans only use 10 percent of their brains."
    python check.py --list                 # show the built-in example claims
    python check.py --example 3            # run example #3
    python check.py --all                  # run every example (slow)

Uses whatever backend .env / the environment selects, so it is free on Ollama.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Optional, Tuple

from fact_checker import Settings, run_multi_agent
from fact_checker.agents import run_baseline
from fact_checker.config import PROVIDERS, ollama_status
from fact_checker.llm import AgentError, ConfigurationError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for _noisy in ("ddgs", "primp", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# (claim, expected verdict, what this example is designed to expose)
EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Humans only use 10 percent of their brains.",
        "REFUTES",
        "Urban myth. Abundant in training data, so an ungrounded model often "
        "repeats it; the web refutes it clearly.",
    ),
    (
        "The Great Wall of China is visible from the Moon with the naked eye.",
        "REFUTES",
        "Another myth with heavy training-data presence. Easy for retrieval.",
    ),
    (
        "The Eiffel Tower is still the tallest structure in the world.",
        "REFUTES",
        "Temporal twist. True in 1889, false since 1930 - needs the word "
        "'still' to be taken seriously.",
    ),
    (
        "Water boils at 100 degrees Celsius at sea level.",
        "SUPPORTS",
        "Stable fact. Both systems should agree with high confidence; a useful "
        "control that the pipeline is not biased toward REFUTES.",
    ),
    (
        "The Burj Khalifa is the tallest building in the world.",
        "SUPPORTS",
        "True but time-sensitive - the kind of fact that silently expires.",
    ),
    (
        "Mount Everest is located entirely within Nepal.",
        "REFUTES",
        "Partially true. The border with China runs across the summit, so a "
        "sloppy check says SUPPORTS.",
    ),
    (
        "Napoleon Bonaparte was unusually short for his time.",
        "REFUTES",
        "Popular misconception rooted in a unit-conversion error.",
    ),
    (
        "The current Secretary-General of the United Nations is Antonio Guterres.",
        "SUPPORTS",
        "Dynamic fact. Goes stale after a training cutoff - exactly where "
        "retrieval beats memory.",
    ),
    (
        "Goldfish have a three-second memory.",
        "REFUTES",
        "Myth refuted by research the model may not recall.",
    ),
    (
        "Bananas grow on trees.",
        "REFUTES",
        "Botanically false (banana plants are herbs) but colloquially assumed "
        "true - tests whether evidence overrides common phrasing.",
    ),
]

_BADGE = {"SUPPORTS": "[+]", "REFUTES": "[-]", "NOT_ENOUGH_INFO": "[?]"}


def resolve_model(settings: Settings) -> Settings:
    """Fall back to an installed local model if the configured one is missing."""
    if not settings.is_local:
        return settings

    reachable, installed = ollama_status(settings.ollama_host)
    if not reachable:
        raise ConfigurationError(
            f"No Ollama server at {settings.ollama_host}.\n"
            "  Install from https://ollama.com, then: ollama pull qwen2.5:3b\n"
            "  If installed, start it with: ollama serve"
        )
    if not installed:
        raise ConfigurationError(
            "Ollama is running but has no models. Run: ollama pull qwen2.5:3b"
        )
    if settings.model not in installed:
        replacement = installed[0]
        print(
            f"  note: {settings.model!r} is not pulled; using {replacement!r} "
            f"(installed: {', '.join(installed)})\n"
        )
        settings.model = replacement
    return settings


def check_claim(claim: str, settings: Settings, *, skip_baseline: bool = False) -> bool:
    """Run both systems over one claim and print a compact comparison.

    Returns:
        True if the run completed, False if a stage failed.
    """
    print("=" * 72)
    print(f"CLAIM: {claim}")
    print("=" * 72)

    try:
        started = time.perf_counter()
        result = run_multi_agent(
            claim, settings, on_progress=lambda msg: print(f"  ... {msg}")
        )
        elapsed = time.perf_counter() - started
    except (AgentError, ConfigurationError) as exc:
        print(f"\n  MULTI-AGENT FAILED: {exc}\n")
        return False

    adjudication = result.adjudication
    print(f"\n  MULTI-AGENT  (grounded, {elapsed:.0f}s)")
    print(f"    queries   : {result.scout.queries}")
    print(f"    evidence  : {len(result.scout.evidence)} snippet(s)")
    for alignment in result.audit.alignments:
        badge = _BADGE.get(alignment.verdict, "[?]")
        print(
            f"      {badge} {alignment.source_id}: {alignment.verdict:<16}"
            f" {alignment.justification[:70]}"
        )
    print(
        f"    VERDICT   : {_BADGE.get(adjudication.final_verdict, '[?]')} "
        f"{adjudication.final_verdict} @ {adjudication.confidence:.0%}"
    )
    print(f"    cited     : {adjudication.cited_sources or '(none)'}")
    print(f"    reasoning : {adjudication.detailed_reasoning[:200]}")
    for warning in result.warnings:
        print(f"    warning   : {warning[:100]}")

    if skip_baseline:
        print()
        return True

    try:
        baseline = run_baseline(claim, settings)
    except (AgentError, ConfigurationError) as exc:
        print(f"\n  BASELINE FAILED: {exc}\n")
        return False

    base = baseline.adjudication
    print(f"\n  BASELINE     (ungrounded, {baseline.seconds:.0f}s)")
    print(
        f"    VERDICT   : {_BADGE.get(base.final_verdict, '[?]')} "
        f"{base.final_verdict} @ {base.confidence:.0%}"
    )
    print(f"    reasoning : {base.detailed_reasoning[:200]}")

    if adjudication.final_verdict != base.final_verdict:
        print("\n  >> VERDICTS DISAGREE - grounded evidence changed the answer.")
    elif base.confidence > adjudication.confidence + 0.15:
        print("\n  >> Same verdict, but the baseline is more confident with no sources.")
    print()
    return True


def list_examples() -> None:
    """Print the example bank with expected verdicts."""
    print("\nBuilt-in examples (run one with: python check.py --example N)\n")
    for index, (claim, expected, why) in enumerate(EXAMPLES, start=1):
        print(f"  {index:>2}. [{expected:<16}] {claim}")
        print(f"      {why}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fact-check a claim from the command line.",
    )
    parser.add_argument("claim", nargs="?", help="The claim to verify.")
    parser.add_argument("--list", action="store_true", help="List example claims.")
    parser.add_argument("--example", type=int, metavar="N", help="Run example N.")
    parser.add_argument("--all", action="store_true", help="Run every example.")
    parser.add_argument("--provider", choices=list(PROVIDERS), default=None)
    parser.add_argument("--model", default=None, help="Override the model id.")
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip the ungrounded control."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        list_examples()
        return 0

    claims: List[str] = []
    if args.all:
        claims = [claim for claim, _, _ in EXAMPLES]
    elif args.example is not None:
        if not 1 <= args.example <= len(EXAMPLES):
            print(
                f"ERROR: --example must be 1..{len(EXAMPLES)}. "
                "Run --list to see them.",
                file=sys.stderr,
            )
            return 1
        claims = [EXAMPLES[args.example - 1][0]]
    elif args.claim:
        claims = [args.claim]
    else:
        list_examples()
        print('Or check your own:  python check.py "your claim here"')
        return 0

    overrides = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model

    try:
        settings = resolve_model(Settings(**overrides))
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nBackend: {settings.provider} / {settings.model}\n")

    failures = 0
    for claim in claims:
        if not check_claim(claim, settings, skip_baseline=args.no_baseline):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
