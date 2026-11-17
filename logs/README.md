# Logs

Output captured from the runs that produced what is in `results/`.

| File | What it is |
|---|---|
| `regenerate.log` | Full console output of `python src/analysis/regenerate.py` — every analysis step, in order, with the figures, tables and macro files it wrote. |
| `pytest.log` | Full output of the 434-test suite. |

Both are real captured runs, not transcripts written by hand.

## What is not here

There is no training log, because nothing is trained — see
`checkpoints/README.md`.

There is no raw retrieval log either. The retrieval phase ran against a
live web search service and was **halted partway through**: the service
began returning empty result sets, and continuing against a failing
endpoint would have filled the passage cache with empty evidence
indistinguishable from genuine "nothing found" results. The run was
stopped at 90 of a planned 250 claims rather than looped. What that phase
produced is preserved in full as structured records in
`cache/av250.jsonl` — one entry per claim, with the queries issued and
every passage returned — which is more faithful than a console log would
have been.

## Regenerating these

    python src/analysis/regenerate.py > logs/regenerate.log 2>&1
    python -m pytest -q > logs/pytest.log 2>&1

Both are deterministic: every sampling step is seeded (`20240101`) and
decoding is greedy, so a rerun reproduces the same numbers.
