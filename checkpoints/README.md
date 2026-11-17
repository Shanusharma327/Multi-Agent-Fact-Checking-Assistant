# Checkpoints

MAFCA is an inference-time pipeline: three agents built around prompts,
typed schemas and deterministic post-processing, all running against a
single frozen off-the-shelf model. No parameter is updated at any point,
there is no training loop in the repository, and there is no optimiser
state, epoch or learning curve to report.

What takes the place of a checkpoint is the exact model the run used:

| | |
|---|---|
| Backend | Ollama (local, no API key) |
| Model tag | `qwen2.5:3b` |
| Digest | `357c53fb659c` |
| Parameters | 3.1 B |
| Quantisation | `Q4_K_M` |
| Context length used | 8192 (`FACTCHECK_NUM_CTX`) |
| Decoding | `temperature=0.0` — greedy, no sampling |

Reproduce that environment with:

    ollama pull qwen2.5:3b

The 1.9 GB weight file is distributed by Ollama and is not mirrored here;
pulling by tag gives the same digest shown above.

## What is actually frozen in this repository

Because the model is fixed and decoding is greedy, the artefacts that make
the results reproducible are not weights but **recorded inputs and
outputs**, and those *are* committed:

| Path | What it pins |
|---|---|
| `cache/av250.jsonl` | Every retrieved passage each condition saw. Retrieval used a live web service whose results change daily, so this file is what makes the experiment replayable at all. |
| `results/av250/` | Per-claim verdicts for every condition. |
| `results/av250_oracle/` | The same claims with the corpus' annotated evidence substituted for web search. |
| `.env.example` | The exact runtime settings the reported run used. |

Together these let `python src/analysis/regenerate.py` rebuild every
reported number with no network access and no model call. See the root
`README.md`.
