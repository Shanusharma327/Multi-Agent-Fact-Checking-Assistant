"""A non-ASCII claim must not end a long run.

The resumed 160-claim retrieval died on its fourth claim: the progress
line echoes the claim text, that claim carried a rupee sign, and a
Windows console defaults to cp1252, which cannot encode it. Three
claims' work was lost with it, the checkpoint interval being ten.

AVeriTeC claims are real-world text, so currency symbols, dashes and
accented names are ordinary rather than exceptional. These tests pin the
guard that keeps the terminal's encoding from deciding whether a run
finishes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ENTRY_POINTS = ["src/run_conditions.py", "src/evaluate.py", "src/run_tier1.py"]

# The character that actually stopped the run, plus the other classes of
# non-ASCII that AVeriTeC claims routinely contain.
HOSTILE = "₹2000 — Laurenço's café – 75°"


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_declares_the_guard(script: str) -> None:
    """Every entry point that prints claim text reconfigures its streams."""
    source = (ROOT / script).read_text(encoding="utf-8")
    assert "_stream.reconfigure" in source, (
        f"{script} prints claim text but does not guard the console encoding"
    )


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_survives_a_cp1252_console(script: str) -> None:
    """Importing the script under cp1252 leaves stdout able to take a rupee sign.

    The guard runs at import, so importing the module under a cp1252
    console and then printing the hostile string reproduces exactly what
    the retrieval run did -- and must now succeed.
    """
    env = dict(os.environ, PYTHONIOENCODING="cp1252",
               PYTHONPATH=str(ROOT / "src"))
    path = ROOT / script
    probe = (
        "import sys\n"
        # Run the module body for its side effects without calling
        # main(); argparse would otherwise exit on the missing
        # subcommand. run_tier1 reads __file__ at module level, so the
        # globals have to carry it.
        f"src = open(r'{path}', encoding='utf-8').read()\n"
        "head = src.split('def main(')[0]\n"
        "exec(compile(head, 'probe', 'exec'),\n"
        f"     {{'__name__': 'probe', '__file__': r'{path}'}})\n"
        f"print({HOSTILE!r})\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert done.returncode == 0, (
        f"{script} crashed under a cp1252 console:\n{done.stderr[-2000:]}"
    )
    assert "UnicodeEncodeError" not in done.stderr


def test_an_unguarded_print_would_have_failed() -> None:
    """The guard is load-bearing, not incidental.

    Without it the same print raises UnicodeEncodeError, which is what
    ended the retrieval run. If this ever stops failing the environment
    has changed and the tests above no longer prove anything.
    """
    env = dict(os.environ, PYTHONIOENCODING="cp1252",
               PYTHONPATH=str(ROOT / "src"))
    done = subprocess.run(
        [sys.executable, "-c", f"print({HOSTILE!r})"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert done.returncode != 0
    assert "UnicodeEncodeError" in done.stderr
