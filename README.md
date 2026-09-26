# 🔎 Multi-Agent Fact-Checking Assistant

A three-agent pipeline that verifies claims against **live web evidence**, running
side by side with a **monolithic single-LLM baseline** that answers from memory
alone. The point of the app is the gap between the two columns: it makes
ungrounded confidence visible.

![status](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/ui-streamlit-red)
![status](https://img.shields.io/badge/tests-143%20passing-brightgreen)

---

## Why a pipeline instead of one prompt?

A single LLM asked "is this true?" answers from parametric memory. It cannot
distinguish *"I recall this"* from *"this is documented"*, so it fails hardest
exactly where users need it most: facts that changed after training, facts it
half-remembers, and popular myths that were abundant in its training data.

Splitting the job into three narrow agents fixes this structurally:

| Failure mode | Structural defence |
| --- | --- |
| Confirmation bias in retrieval | Agent 1 is forbidden from judging the claim, and must write one query from an **independent angle** |
| Model's memory leaking into "evidence" | Agent 2 runs under a **closed-world rule** — only the snippet's own text may justify a label |
| Verdict drowning in weak corroboration | Agent 3 weighs **specificity over volume** and must justify what it discounted |
| Fabricated citations | Cited IDs are **validated against real retrieved evidence** in code, not trusted from the model |

---

## Architecture

```
                 claim
                   │
     ┌─────────────▼──────────────┐
     │  AGENT 1 · Evidence Scout  │  2 neutral queries → DuckDuckGo
     │  → queries[], evidence[]   │  → snippets as doc_1 … doc_n
     └─────────────┬──────────────┘
                   │
     ┌─────────────▼──────────────┐
     │ AGENT 2 · Alignment Auditor│  closed-world NLI, one label per snippet
     │  → alignments[]            │  SUPPORTS / REFUTES / NOT_ENOUGH_INFO
     └─────────────┬──────────────┘
                   │
     ┌─────────────▼──────────────┐
     │ AGENT 3 · Synthesis Adjud. │  weighs conflicts, calibrates confidence
     │  → final verdict + cites   │
     └─────────────┬──────────────┘
                   │
                   ▼                        ┌───────────────────────────┐
           grounded verdict   ◄── compare ──►│ BASELINE · single LLM     │
                                             │ zero-shot, no search      │
                                             │ identical output schema   │
                                             └───────────────────────────┘
```

Both paths emit the **same** `Adjudication` schema. That is what makes the
comparison 1:1 rather than impressionistic.

### Output schemas

```python
ScoutReport   { queries: List[str], evidence: List[{source_id, snippet, title, url, query}] }
AuditReport   { alignments: List[{source_id, verdict, justification}] }
Adjudication  { claim, final_verdict, confidence: float, detailed_reasoning, cited_sources: List[str] }
```

`title`, `url` and `query` on evidence items are display-only retrieval metadata
— the agents themselves only ever reason over `source_id` and `snippet`.

---

## Quickstart

```bash
git clone <your-repo> && cd proj

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then pick a backend. **No API key is required** — the whole pipeline, including
the FEVER benchmark, runs on a local model.

### Option A — Local model, free (default)

```bash
# 1. Install Ollama from https://ollama.com
# 2. Pull a model (~4.7 GB):
ollama pull qwen2.5:7b
# 3. Run:
streamlit run app.py
```

Inference runs on your machine: no account, no key, no per-call cost. Web
search still uses the internet.

| Model | Size | VRAM | Notes |
| --- | --- | --- | --- |
| `qwen2.5:7b` | 4.7 GB | ~6 GB | **Default.** Strongest JSON adherence in its class |
| `llama3.1:8b` | 4.9 GB | ~6 GB | Good general reasoning |
| `qwen2.5:3b` | 1.9 GB | ~4 GB | Faster; use if 7B is tight or slow |
| `gemma2:2b` | 1.6 GB | ~3 GB | Fastest, weakest reasoning |

### Option B — Hosted Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env, or paste in the sidebar
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows which backend is live and
whether it is reachable; all configuration lives in `.env` (copy
`.env.example`) so switching backends is a one-line edit plus a restart.

Backend resolution order: `FACTCHECK_PROVIDER` env var → `anthropic` if
`ANTHROPIC_API_KEY` is set → otherwise `ollama`.

### Examples you can run

`check.py` verifies a single claim from the terminal, no browser needed:

```bash
python check.py --list                  # the example bank, with expected verdicts
python check.py --example 2             # run example #2
python check.py "your own claim here"   # check anything
python check.py --all                   # run all 10 (slow)
python check.py --no-baseline           # grounded path only, ~2x faster
```

The bank is chosen so each example probes a different failure mode:

| # | Claim | Expected | Probes |
| --- | --- | --- | --- |
| 1 | Humans only use 10% of their brains | `REFUTES` | Myth abundant in training data |
| 2 | Great Wall visible from the Moon | `REFUTES` | Same, easy for retrieval |
| 3 | Eiffel Tower **still** the tallest | `REFUTES` | Temporal twist — true in 1889 |
| 4 | Water boils at 100 °C at sea level | `SUPPORTS` | Control: not biased toward REFUTES |
| 5 | Burj Khalifa is the tallest building | `SUPPORTS` | True, but silently expires |
| 6 | Everest lies entirely within Nepal | `REFUTES` | Partially true — border runs the summit |
| 7 | Napoleon was unusually short | `REFUTES` | Misconception from a unit-conversion error |
| 8 | Guterres is current UN Secretary-General | `SUPPORTS` | Dynamic fact, stale after cutoff |
| 9 | Goldfish have a three-second memory | `REFUTES` | Myth refuted by research |
| 10 | Bananas grow on trees | `REFUTES` | Botanically false, colloquially assumed true |

### Using it

1. Type a claim, or hit a **quick-load example** in the sidebar:
   - 🕰️ **Temporal twist** — true once, quietly false now
   - 🧪 **Urban myth** — widely repeated, refuted by evidence
   - 📈 **Recent dynamic fact** — the kind that goes stale in parametric memory
2. Press **Verify Claim**.
3. Watch the left column trace each agent live; read the baseline on the right.
4. The **Head-to-head** panel below calls out disagreement and baseline
   overconfidence explicitly.

---

## Deployment

### Docker

```bash
docker build -t factcheck .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8501:8501 factcheck
```

The key is injected at run time and never baked into the image. The container
ships a healthcheck against Streamlit's `/_stcore/health` endpoint.

### Google Colab

```python
!git clone <your-repo> && %cd proj
!pip install -q -r requirements.txt pyngrok

import colab_setup
print(colab_setup.launch("sk-ant-...", ngrok_token="<your-ngrok-token>"))
```

### Headless / scripted use

The pipeline is importable and has no Streamlit dependency:

```python
from fact_checker import Settings, run_multi_agent
from fact_checker.agents import run_baseline

settings = Settings(api_key="sk-ant-...")          # or rely on the environment
result   = run_multi_agent("Humans only use 10% of their brains.", settings)

print(result.adjudication.final_verdict)            # REFUTES
print(result.adjudication.confidence)               # e.g. 0.92
print(result.adjudication.cited_sources)            # ['doc_1', 'doc_4']
print(result.total_seconds)

baseline = run_baseline("Humans only use 10% of their brains.", settings)
print(baseline.adjudication.final_verdict)
```

---

## Benchmarks (FEVER + AVeriTeC)

The Streamlit dashboard gives a *qualitative* per-claim comparison. For numbers,
`evaluate.py` scores both systems over [FEVER](https://fever.ai) — 185k claims
labelled `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`, the same three-way label
set this pipeline emits, which is what makes it a drop-in ground truth.

```bash
python evaluate.py --download                      # FEVER dev (~4.3 MB)
python evaluate.py --dataset averitec --download   # AVeriTeC dev (~1.8 MB)

python evaluate.py --n 10 --dry-run                # preview sample + cost
python evaluate.py --n 10                          # FEVER (comparability)
python evaluate.py --dataset averitec --n 10       # AVeriTeC (validity)
python evaluate.py --score-only                    # re-score existing results
```

**Two corpora, two purposes.** FEVER is the canonical benchmark, so its numbers are
comparable to prior work — but it was published in 2018 (so it may be *in* the models'
training data, biasing the ungrounded control), its NEI label is corpus-relative, and its
claims are mutated Wikipedia sentences. AVeriTeC is real-world, open-web, 2023, and
matches what this system actually targets. Run both; they are expected to disagree.

`--n` is **per label**, so `--n 10` runs 30 claims. Results stream to
`results/fever_run.jsonl` as they complete; `--resume` skips claims already
recorded rather than paying for them twice.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--n` | `10` | Claims per label (×3) |
| `--split` | `dev` | `dev`, `train`, `paper_dev`, `paper_test` |
| `--seed` | `20240101` | Sampling seed — same seed, same claims |
| `--model` | `claude-opus-5` | Also `claude-sonnet-5`, `claude-haiku-4-5` |
| `--workers` | `3` | Parallel claims (keep low; DuckDuckGo throttles) |
| `--resume` | off | Skip claims already in the output file |
| `--yes` | off | Skip the cost confirmation |

### What it measures

Accuracy and macro-F1, **plus calibration** — because accuracy alone cannot show
the thing this project is about. Two systems with the same accuracy are not
equivalent if one of them says 95% when it is wrong.

- **ECE** (Expected Calibration Error) — gap between stated confidence and
  observed accuracy, bucketed. Lower is better.
- **Brier score** — squared error between confidence and correctness.
- **Overconfidence** — `mean_confidence − accuracy`. Positive means the system
  claims more certainty than it earns.
- **Confidence when right vs. when wrong** — a well-behaved system separates
  these; an ungrounded one does not.

Sampling is label-balanced by default. The `dev` split is already balanced by
construction (6,666 claims per label — see `analysis/dataset_stats.py`), so this
is a no-op at full size; it matters at the small sample sizes actually used,
where an unstratified draw fluctuates enough to skew accuracy by chance, and for
splits that carry no such guarantee.

### ⚠️ How to read a FEVER score here

**FEVER's labels are defined against a June 2017 Wikipedia dump. This pipeline
retrieves from the open web.** That mismatch is deliberate but it changes the
interpretation, and the harness prints the caveat with every report:

- `NOT ENOUGH INFO` means *"unverifiable from that Wikipedia snapshot"*. Live
  search can legitimately find evidence for such a claim, so the pipeline may
  answer `SUPPORTS` where FEVER says `NEI` — and be **right about the world**
  while scoring as wrong. Expect depressed NEI recall.
- FEVER claims are decontextualised mutations of Wikipedia sentences ("Tilda
  Swinton is a vegan."), often about obscure entities with ambiguous referents.
- This is therefore **"FEVER-labelled claims under open-web retrieval"**, not
  official FEVER scoring — the official metric additionally requires retrieving
  the gold Wikipedia sentences, which this pipeline never sees.

What it *does* support cleanly is the comparison this project exists for: both
systems face identical claims under identical conditions.

### Cost

Four LLM calls per claim (3 pipeline + 1 baseline).

**Running locally costs nothing**, so benchmark size is limited only by
patience — expect roughly 1–3 minutes per claim for a 7B model on a mid-range
GPU, most of it in the auditor stage:

```bash
python evaluate.py --provider ollama --model qwen2.5:7b --n 10
```

For the hosted backend, `--dry-run` prints an estimate and the harness asks for
confirmation before spending:

| Model | ~per claim | `--n 10` (30 claims) |
| --- | --- | --- |
| `qwen2.5:7b` (local) | **$0.00** | **$0.00** |
| `claude-haiku-4-5` | ~$0.02 | ~$0.47 |
| `claude-sonnet-5` | ~$0.03 | ~$0.93 |
| `claude-opus-5` | ~$0.08 | ~$2.30 |

Start with `--n 2` to shake out the plumbing before committing to a long run.

> **A caution when benchmarking a small local model.** A 7B model is markedly
> worse at the *baseline* task than Claude is, so the grounded-vs-ungrounded gap
> will usually look **larger** locally. That is a property of the weaker model,
> not extra evidence that retrieval helps. Compare local-to-local and
> hosted-to-hosted; do not read a local gap as a hosted one.

---

## Configuration

| Setting | Env var | Default | Notes |
| --- | --- | --- | --- |
| Backend | `FACTCHECK_PROVIDER` | auto | `ollama` or `anthropic`; auto-detects from key presence |
| Model | `FACTCHECK_MODEL` | per backend | `qwen2.5:7b` local, `claude-opus-5` hosted |
| API key | `ANTHROPIC_API_KEY` | — | Hosted backend only; sidebar overrides the env var |
| Ollama host | `OLLAMA_HOST` | `http://localhost:11434` | Point at a remote box if you have one |
| Context window | `FACTCHECK_NUM_CTX` | `8192` | Local only. **Do not lower** — see below |
| Results per query | `FACTCHECK_RESULTS_PER_QUERY` | `4` | Sidebar slider, 2–8 |
| Max evidence snippets | `FACTCHECK_MAX_EVIDENCE` | `8` | Sidebar slider; defaults to 4 on local |

> **Two backend-specific gotchas, both handled in `fact_checker/llm.py`:**
>
> **Anthropic — no sampling parameters.** `build_llm()` passes no
> `temperature` / `top_p` / `top_k`; the Opus 5 generation rejects them with a
> 400. Determinism comes from strict schemas and closed-world prompts instead.
>
> **Ollama — `num_ctx` must be set explicitly.** Ollama defaults to a
> **2048-token** context and *silently truncates* anything longer. The auditor
> prompt carrying several snippets blows straight past that, which would look
> like the model "ignoring evidence" rather than an error. It is pinned to 8192,
> and `temperature=0` is set for reproducibility (unlike Opus, Ollama accepts it).

---

## Engineering notes

Things that are easy to get wrong in a multi-agent app, and how they are handled:

**Search is treated as unreliable by default.** DuckDuckGo throttles without
warning. `fact_checker/search.py` retries with exponential backoff (doubling the
delay when the error looks like rate limiting) and, on total failure, returns an
empty evidence list plus a human-readable warning rather than raising. The run
continues and correctly concludes `NOT_ENOUGH_INFO` instead of crashing the UI.

**One snippet cannot masquerade as several.** Results are de-duplicated by URL
*and* by the opening of the snippet body, so a story syndicated across mirrors
does not read as independent corroboration.

**Agent 2's output is reconciled, not trusted.** `_reconcile()` enforces a
one-to-one mapping between snippets and alignments: hallucinated `doc_N` IDs are
dropped, omitted snippets are backfilled as `NOT_ENOUGH_INFO`, and duplicate
judgements keep the first. Agent 3 therefore always receives a complete audit.

**Citations are validated against reality.** `_sanitise_citations()` discards any
cited ID that does not correspond to a genuinely retrieved snippet. The baseline's
citations are discarded wholesale — it has no sources, so anything it cites is
fabricated by construction.

**The claim is pinned to the user's wording.** Both paths overwrite the model's
restated `claim` field with the exact input, so a paraphrase cannot quietly
change what the verdict is about.

**Confidence is clamped, not rejected.** A model returning `85` (a percentage) is
rescaled to `0.85`; a model returning `1.4` (an overshoot) is clamped to `1.0`.
Treating the overshoot as a percentage would silently turn "certain" into 1.4%.

---

## Report

A full technical write-up — problem statement, introduction, related work,
methodology with algorithms and mathematical formulation, and dataset analysis —
is in **[REPORT.md](REPORT.md)**. Dataset tables (3-8) and figures (1-4) are
regenerated from the cached corpora by:

```bash
python analysis/dataset_stats.py                  # both corpora
python analysis/dataset_stats.py --download       # fetch anything not cached
python analysis/dataset_stats.py --dataset fever  # one corpus only
```

Outputs: `analysis/figures/*.png` and `analysis/dataset_stats.json`. No number in
the report is hard-coded; all are computed by this script.

---

## Project layout

```
proj/
├── app.py                        # Streamlit dashboard
├── check.py                      # One-claim CLI + example bank
├── evaluate.py                   # FEVER benchmark CLI
├── colab_setup.py                # One-cell Colab launcher
├── Dockerfile
├── requirements.txt
├── .env.example
├── fact_checker/
│   ├── config.py                 # Settings: args → env → defaults
│   ├── schemas.py                # Pydantic v2 contracts (shared by both paths)
│   ├── search.py                 # DuckDuckGo retrieval, backoff, dedup
│   ├── pipeline.py               # Stage-by-stage orchestration
│   ├── llm.py                    # Backend factory (Ollama / Anthropic)
│   ├── metrics.py                # Accuracy, macro-F1, ECE, Brier
│   ├── agents/
│   │   ├── scout.py              # Agent 1
│   │   ├── auditor.py            # Agent 2
│   │   ├── adjudicator.py        # Agent 3
│   │   └── baseline.py           # Monolithic control
│   └── datasets/
│       ├── base.py               # Shared BenchmarkClaim + stratified sampler
│       ├── fever.py              # FEVER loader (comparability)
│       └── averitec.py           # AVeriTeC loader (validity)
├── analysis/
│   ├── dataset_stats.py          # Tables 3-8, Figures 1-4
│   └── figures/                  # Generated PNGs
├── data/                         # Downloaded corpora (gitignored)
├── results/                      # Benchmark output JSONL (gitignored)
└── tests/                        # 143 offline tests, no network, key or model
```

---

## Tests

```bash
pip install pytest
pytest tests/ -q
```

The suite is fully offline — it stubs the search backend and the LLM, and never
calls the Anthropic API or the network. It covers the deterministic glue that
protects the verdict from a sloppy model response: confidence coercion, audit
reconciliation, citation sanitising, snippet de-duplication, truncation, and
graceful rate-limit degradation.

`tests/test_end_to_end.py` runs the real Agent 1 → 2 → 3 orchestration against a
stubbed model that deliberately misbehaves — it hallucinates a `doc_99`, skips a
snippet, cites a non-existent `doc_77`, and paraphrases the claim — and asserts
that every one of those is caught before it reaches the verdict.

`tests/test_evaluate.py` drives the FEVER harness the same way: real sampling,
threading, JSONL writes, resume and scoring, with the model stubbed out. The
metrics themselves are pinned against hand-computed values in
`tests/test_metrics.py` (a perfectly calibrated system scores ECE 0.0; a
confidently wrong one scores 1.0).

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No Anthropic API key found` | Set `ANTHROPIC_API_KEY`, paste a key into the sidebar, or switch the backend to Ollama |
| `Ollama not reachable` | Install from [ollama.com](https://ollama.com) and check `ollama list` |
| `model ... is not pulled` | `ollama pull qwen2.5:7b` |
| Local run is very slow | Drop to `qwen2.5:3b`, or lower **Max evidence snippets** to 3–4 |
| Local model returns malformed JSON | It retries 3x automatically; if it persists use `qwen2.5:7b` (best JSON adherence) or raise `FACTCHECK_NUM_CTX` |
| `No DuckDuckGo client installed` | `pip install ddgs` |
| Every verdict is `NOT_ENOUGH_INFO` | Retrieval is likely throttled — check the *Stage timings & diagnostics* expander for warnings, then retry after a minute |
| Emoji render as `?` in your terminal | Cosmetic console encoding only; the browser UI is unaffected |

---

## License

MIT.
