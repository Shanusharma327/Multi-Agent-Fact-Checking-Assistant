# Data

## Not committed

The raw benchmark corpora are **not** in this repository. Fetch them with:

    python data/download_datasets.py

| Corpus | Files | Size | Source |
|---|---|---|---|
| AVeriTeC | `averitec/dev.json`, `averitec/train.json` | ~12 MB | <https://github.com/MichSchli/AVeriTeC> |
| FEVER *(optional)* | `fever/shared_task_dev.jsonl` | ~4 MB | <https://fever.ai/> |

Each is redistributed by its own authors under its own licence, so this
repository links and downloads rather than mirrors them.

FEVER is optional and not reported: a 30-claim pilot could not support the
cross-corpus claim it was being used for, so it was withdrawn. Pass
`--fever` to fetch it anyway.

## Why AVeriTeC

Every annotated answer carries a **source URL**, which makes retrieval
quality measurable on its own terms rather than inferred backwards from
the final verdict — unlike benchmarks whose gold evidence is a sentence
index into a fixed Wikipedia snapshot.

## You do not need this to reproduce the results

Downloading the corpora is only required to draw a **new** sample of
claims. Every reported number is rebuilt from committed files:

- `cache/av250.jsonl` — the retrieved passages every condition saw
- `results/av250/`, `results/av250_oracle/` — per-claim verdicts

so `python src/analysis/regenerate.py` runs offline, with no corpus
download and no model call.
