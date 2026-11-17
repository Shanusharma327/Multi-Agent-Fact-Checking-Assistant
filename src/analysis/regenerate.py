"""Regenerate every table, figure and macro in ``results/`` from the run files.

One entry point, so the reported artefacts cannot be partially rebuilt. It
runs no inference and touches no network: everything here is derived from
the stored per-claim output under ``results/``, which in turn replays the
passage cache in ``cache/``. Every step is seeded, so two runs of this
script on the same inputs produce the same numbers.

    python src/analysis/regenerate.py

Outputs, all under ``results/``:

    results/figures/      every chart the report uses
    results/tables/       every LaTeX table
    results/macros/       numbers.tex, condition_numbers.tex,
                          attribution_numbers.tex -- the generated
                          quantities, one macro per reported figure

Step order matters in one place: ``condition_tables.py`` truncates
``condition_numbers.tex`` and ``oracle_macros.py`` appends to it, so
running them out of order duplicates every appended macro.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/analysis/..)
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
MACRO_DIR = RESULTS / "macros"


def run(args: Sequence[str]) -> bool:
    """Run one generation step, reporting rather than raising on failure."""
    printable = " ".join(str(a) for a in args[1:])
    print(f"\n$ python {printable}")
    result = subprocess.run([sys.executable, *[str(a) for a in args[1:]]], cwd=ROOT)
    if result.returncode != 0:
        print(f"  ! step failed with exit code {result.returncode}", file=sys.stderr)
        return False
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--averitec", type=Path, default=RESULTS / "averitec_n90_v2.jsonl"
    )
    # FEVER is no longer reported: a 30-claim pilot could not support
    # the cross-corpus claim it was used for, so the report withdrew it.
    # Pass --fever explicitly to rebuild those artefacts anyway.
    parser.add_argument("--fever", type=Path, default=None)
    # The \Or macro family came from the superseded 60-claim oracle run
    # and nothing in the manuscript cited it; the oracle analysis the
    # report gives uses \Cond...Oracle..., emitted by oracle_macros.py
    # from results/av250_oracle. Emitting 216 unused macros off an old
    # run invites someone to quote one, so it is off by default.
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--betas", type=float, nargs="*", default=[0.0, 1.0, 2.0])
    parser.add_argument(
        "--skip-join", action="store_true",
        help="Keep the existing paired results file instead of rebuilding it.",
    )
    args = parser.parse_args(argv)

    for directory in (FIG_DIR, TAB_DIR, MACRO_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    live = RESULTS / "av250"
    oracle_dir = RESULTS / "av250_oracle"
    cache = ROOT / "cache" / "av250.jsonl"

    if not live.exists():
        print(f"Missing condition output: {live}", file=sys.stderr)
        return 1

    ok = True

    # 1. Reshape the per-condition files into the paired record the
    #    reporters read, so both halves of the results describe one run.
    if not args.skip_join:
        ok &= run(["python", "src/analysis/join_conditions.py",
                   "--conditions", live, "--cache", cache, "--out", args.averitec])

    if not args.averitec.exists():
        print(f"Missing results file: {args.averitec}", file=sys.stderr)
        return 1

    # 2. Per-corpus reports, figures and the aggregation ablation.
    corpora = [(args.averitec, "av")]
    if args.fever:
        corpora.append((args.fever, "fev"))
    for results, tag in corpora:
        ok &= run(["python", "src/analysis/results_report.py",
                   "--results", results, "--tag", tag])
        ok &= run(["python", "src/analysis/grounding_figures.py", "--results", results,
                   "--tag", tag, "--betas", *[str(b) for b in args.betas]])
        ok &= run(["python", "src/analysis/ablation.py", "--results", results,
                   "--tag", tag, "--betas", *[str(b) for b in args.betas]])

    ok &= run(["python", "src/analysis/summary_figures.py"])

    # 3. The \Av macro family.
    macro_args = ["python", "src/analysis/paper_numbers.py",
                  "--averitec", args.averitec]
    if args.fever:
        macro_args += ["--fever", args.fever]
    if args.oracle and args.oracle.exists():
        macro_args += ["--oracle", args.oracle]
    macro_args += ["--out", MACRO_DIR / "numbers.tex"]
    ok &= run(macro_args)

    # 4. The \Cond macro family and the condition tables. This truncates
    #    condition_numbers.tex, so it must precede oracle_macros.py.
    cond_macros = MACRO_DIR / "condition_numbers.tex"
    ok &= run(["python", "src/analysis/condition_tables.py", "--dir", live,
               "--cache", cache, "--out-dir", TAB_DIR, "--macros", cond_macros])

    # 5. Oracle, sensitivity and power macros, appended to the same file.
    if oracle_dir.exists():
        ok &= run(["python", "src/analysis/oracle_macros.py", "--live", live,
                   "--oracle", oracle_dir, "--cache", cache, "--out", cond_macros])
    else:
        print(f"\n  - skipping oracle_macros.py: {oracle_dir} not present")

    # 6. Error attribution with bootstrap intervals.
    ok &= run(["python", "src/analysis/attribution_ci.py",
               "--out", MACRO_DIR / "attribution_numbers.tex"])

    # 7. The standalone accuracy chart.
    ok &= run(["python", "src/analysis/poster_figures.py"])

    if not ok:
        print("\nAt least one step failed; artefacts may be incomplete.", file=sys.stderr)
        return 1

    n_fig = len(list(FIG_DIR.glob("*.png")))
    n_tab = len(list(TAB_DIR.glob("*.tex")))
    n_mac = len(list(MACRO_DIR.glob("*.tex")))
    print(f"\nAll artefacts regenerated: {n_fig} figure(s) -> {FIG_DIR}")
    print(f"                           {n_tab} table(s)  -> {TAB_DIR}")
    print(f"                           {n_mac} macro file(s) -> {MACRO_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
